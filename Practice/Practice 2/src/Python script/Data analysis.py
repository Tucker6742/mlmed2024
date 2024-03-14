# %% [markdown]
# # $\text{Import library}$

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from varname import argname
import pathlib
from pathlib import Path
from PIL import Image
import numpy.linalg as npla
from matplotlib.patches import Ellipse
from tqdm.contrib.concurrent import thread_map
from matplotlib.patches import Polygon

# %% [markdown]
# # $\text{Utils functions}$


# %%
def get_ellipse_param(anno_path: Path | str, mode="xyabt") -> list[str | float]:
    """
    - Parameters
        - anno_path: Path (path to the annotation image)

    - Behavior

        Given an annotation image, this function returns the parameters of the ellipse that best fits the annotation.

    - Returns
        - list[str | float]: [file_name, x_center, y_center, width, height, angle]
    """
    if isinstance(anno_path, str):
        anno_path = Path(anno_path)
    anno = Image.open(anno_path)
    pixel_array = np.array(anno.getdata()).reshape(anno.size[1], anno.size[0])
    coords = np.array(np.where(pixel_array))[::-1]
    coef_matrix = np.array(
        [coords[0] ** 2, coords[0] * coords[1], coords[1] ** 2, coords[0], coords[1]],
        dtype=int,
    ).T
    ones_matrix = np.ones_like(coef_matrix[:, 0])
    parameter = npla.lstsq(coef_matrix, ones_matrix, rcond=None)[0]
    A, B, C, D, E, F = parameter.tolist() + [-1]

    a = -np.sqrt(
        2
        * (A * E**2 + C * D**2 - B * D * E + F * (B**2 - 4 * A * C))
        * ((A + C) + np.sqrt((A - C) ** 2 + B**2))
    ) / (B**2 - 4 * A * C)

    b = -np.sqrt(
        2
        * (A * E**2 + C * D**2 - B * D * E + F * (B**2 - 4 * A * C))
        * ((A + C) - np.sqrt((A - C) ** 2 + B**2))
    ) / (B**2 - 4 * A * C)
    x0 = (2 * C * D - B * E) / (B**2 - 4 * A * C)
    y0 = (2 * A * E - B * D) / (B**2 - 4 * A * C)
    theta = 0.5 * np.arctan2(-B, (C - A)) * 180 / np.pi

    name = anno_path.name.rsplit("_", 1)[0] + ".png"
    if mode == "xyabt":
        return [name, x0, y0, a, b, theta]
    elif mode == "abcdef":
        return [name, A, B, C, D, E, F]


# %%
def get_obb_point(x, y, a, b, t, image_path, mode="xyxyxyxy") -> list[str | float]:
    """
    - Parameters
        - anno_path: Path (path to the annotation image)
        - mode: str (output format)

    - Behavior
        Given an annotation image, this function returns the parameters of the oriented bounding box that best fits the annotation.

        A: rightmost point after rotation
        B: top point after rotation
        C: leftmost point after rotation
        D: bottom point after rotation
        X: center of the ellipse

        O1: bbox point between A, B
        O2: bbox point between B, C
        O3: bbox point between C, D
        O4: bbox point between D, A

        OO1 = OA + AO1 = OA + XB
        OO2 = OB + BO2 = OB + XC
        OO3 = OC + CO3 = OC + XD
        OO4 = OD + DO4 = OD + XA
    - Returns
        - list[str | float]: [file_name, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    image = Image.open(image_path)
    A = np.array([x + a * np.cos(t / 180 * np.pi), y + a * np.sin(t / 180 * np.pi)])
    B = np.array([x - b * np.sin(t / 180 * np.pi), y + b * np.cos(t / 180 * np.pi)])
    C = np.array([x - a * np.cos(t / 180 * np.pi), y - a * np.sin(t / 180 * np.pi)])
    D = np.array([x + b * np.sin(t / 180 * np.pi), y - b * np.cos(t / 180 * np.pi)])
    X = np.array([x, y])

    x1, y1 = A + B - X
    x2, y2 = B + C - X
    x3, y3 = C + D - X
    x4, y4 = D + A - X

    x_val = np.array([x1, x2, x3, x4]) / image.size[0]
    y_val = np.array([y1, y2, y3, y4]) / image.size[1]

    x1, x2, x3, x4 = x_val
    y1, y2, y3, y4 = y_val

    if mode == "xyxyxyxy":
        return [x1, y1, x2, y2, x3, y3, x4, y4]


# %%
def formatting_data(
    data_path: pathlib.Path, root_dir: Path, *args, **kwargs
) -> pl.DataFrame:
    """

    - Parameters:

        - `data_path`: pathlib.Path (path to the train folder)
        - `root_dir`: pathlib.Path (path to the root directory)


    - Behavior:


        Create `data` dataframe from image name, image path, `data_pixel` from image name, pixel size

        It make `data_param` from the annotation path, and join `data` and `data_param` on `image_name` to get the ellipse parameters. Add the annotation path and the head circumference to the `data` dataframe.


    - Returns:

        `data`: `pl.DataFrame `


        (dataframe containing the image aname, image path, annotation path, and the ellipse parameters, the pixel size and the head circumference) if the data_path is the train folder.)


    """
    name = data_path.name
    print(f"Formatting {name} data")
    annotation_paths = sorted((data_path / "annotations").glob("*.png"))
    image_paths = sorted((data_path / "images").glob("*.png"))
    data = pl.DataFrame(
        {
            "image_name": [i.name for i in image_paths],
            "image_path": [str(i) for i in image_paths],
            "annotation_path": [str(i) for i in annotation_paths],
        }
    )

    data_pixel = pl.read_csv(
        root_dir / "training_set_pixel_size_and_HC.csv",
        new_columns=["image_name", "pixel_size", "head_circumference"],
    )

    data = data.join(data_pixel, on="image_name")

    ellipsis_param = thread_map(get_ellipse_param, data["annotation_path"].to_list())
    data_param = pl.DataFrame(
        ellipsis_param,
        schema=["image_name", "x0", "y0", "a", "b", "theta"],
    )

    data = data.join(data_param, on="image_name")

    data = data.select(
        [
            "image_name",
            "image_path",
            "annotation_path",
            "x0",
            "y0",
            "a",
            "b",
            "theta",
            "pixel_size",
            "head_circumference",
        ]
    )

    return data


# %%
def write_an_obb(row_dict: dict[str, str | float]) -> None:
    """
    - Parameters
        - row_dict: dict (row dictionary of a dataframe)
        - label_path: Path (path to the label directory)

    - Behavior

        Given a row dictionary, this function writes the oriented bounding box of the annotation image.

        class: 0:head
        x1, y1: highest point
        x2, y2: rightmost point
        x3, y3: lowest point
        x4, y4: leftmost point

    - Returns
        - None
    """
    x, y, a, b, t = (
        row_dict["x0"],
        row_dict["y0"],
        row_dict["a"],
        row_dict["b"],
        row_dict["theta"],
    )
    image_path = row_dict["image_path"]
    name = row_dict["image_name"].replace(".png", ".txt")
    obb_point = get_obb_point(x, y, a, b, t, image_path)
    row = [0] + obb_point
    string_insert = f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} {row[6]} {row[7]} {row[8]}"
    path_label = Path(row_dict["image_path"]).parents[1] / "labels" / name
    with open(path_label, "w") as file:
        file.write(string_insert)


# %%
def write_oob_labels(data: pl.DataFrame, root_dir: Path) -> None:
    """
    - Parameters
        - anno_root_dir: Path (path to the annotation directory)
        - root_dir: Path (path to the root directory)

    - Behavior

        Given an annotation directory, this function writes the parameters of the oriented bounding box that best fits the annotation to each text file using multi-threading.

    - Returns
        - None
    """
    name = argname("data").split("_")[1]
    print(f"Writing oriented bounding box labels to {root_dir  / name / 'labels'} ...")
    row_dicts = data.to_dicts()
    thread_map(write_an_obb, row_dicts)


# %%
def plot_ellipse_sample_data(
    file_names: list[str], data: pl.DataFrame, df_name: str, *args, **kwargs
) -> None:
    """
    - Parameters
        - file_names: list[str] (list of file names)
        - data: pl.DataFrame (dataframe containing the ellipse parameters)
        - df_name: str (name of the dataframe)

    - Behavior
        Given a list of file names and a dataframe, this function plots the ellipse that best fits the annotation image.

    - Returns
        - None
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    ax = ax.flatten()
    for i, file_name in enumerate(file_names):
        row = data.filter(data["image_name"] == file_name)
        x, y, a, b, t = (
            row["x0"][0],
            row["y0"][0],
            row["a"][0],
            row["b"][0],
            row["theta"][0],
        )
        image = Image.open(row["image_path"][0])
        ax[i].imshow(image, cmap="gray")
        ellipse = Ellipse((x, y), 2 * a, 2 * b, angle=t, fill=False, color="r")
        ax[i].add_patch(ellipse)
        ax[i].set_title(file_name)

    fig.suptitle(f"Ellipse samples from {df_name} data")
    plt.show()


# %%
def plot_rectangle_sample_data(
    file_name: list[str], data: pl.DataFrame, df_name: str, *args, **kwargs
) -> None:
    """
    - Parameters:
        - `file_name`: list[str] (list of image names)
        - `df_name`: str (name of the dataframe)
        - `root_dir`: Path (path to the root directory)

    - Behavior:

            Plot some sample images with their corresponding oriented bounding box annotation.

    - Returns:
        None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        image_path = Path(
            data.filter(pl.col("image_name") == file_name[i])["image_path"].item()
        )
        image = Image.open(image_path)
        ax.imshow(image, cmap="gray")
        with open(
            image_path.parents[1] / "labels" / file_name[i].replace("png", "txt")
        ) as f:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, f.read().split()[1:])
        X = np.array([x1, x2, x3, x4]) * image.size[0]
        Y = np.array([y1, y2, y3, y4]) * image.size[1]
        x1, x2, x3, x4 = X
        y1, y2, y3, y4 = Y
        rec = Polygon(
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rec)
        ax.set_title(file_name[i])
    fig.suptitle(
        f"Sample images in {df_name} with their corresponding oriented bounding box annotation"
    )
    plt.show()


# %%
def plot_data(data: pl.DataFrame, root_dir: Path, *args, **kwargs) -> None:
    """
    - Parameters:
        - `data`: pl.DataFrame (dataframe containing the image aname, image path, annotation path, and the ellipse parameters, the pixel size and the head circumference)
        - `root_dir`: Path (path to the root directory)

    - Behavior:
        Plot some sample images with their corresponding ellipse and oriented bounding box annotation.

    - Returns:
        None
    """

    df_name = argname("data").split("_")[1]

    file_names = data["image_name"][:4].to_list()
    plot_ellipse_sample_data(file_names, data, df_name)

    plot_rectangle_sample_data(file_names, data, df_name)


# %% [markdown]
# # $\text{Reads and Analyze data}$
if __name__ == "__main__":
    # %%
    root_dir = Path("../../data")

    # %%
    data_path_train: pathlib.Path = Path("../../data/train")
    data_train: pl.DataFrame = formatting_data(data_path_train, root_dir)
    data_train.head()

    # %%
    data_path_val = Path("../../data/val")
    data_val: pl.DataFrame = formatting_data(data_path_val, root_dir)
    data_val.head()

    # %%
    data_path_test = Path("../../data/test")
    data_test: pl.DataFrame = formatting_data(data_path_test, root_dir)
    data_test.head()

    # %% [markdown]
    # # $\text{Make OBB labels}$

    # %%
    write_oob_labels(data_train, root_dir)
    write_oob_labels(data_val, root_dir)
    write_oob_labels(data_test, root_dir)

    # %% [markdown]
    # # $\text{Plot data}$

    # %%
    plot_data(data_train, root_dir)

    # %%
    plot_data(data_val, root_dir)

    # %%
    plot_data(data_test, root_dir)
