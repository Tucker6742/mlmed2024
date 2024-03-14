# %% [markdown]
# # $\text{Import library}$

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import polars.selectors as cs
from varname import argname
import pathlib
from pathlib import Path
from PIL import Image
import numpy.linalg as npla
from matplotlib.patches import Ellipse
from tqdm.contrib.concurrent import thread_map

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
    # a, b = max(a, b), min(a, b)
    x0 = (2 * C * D - B * E) / (B**2 - 4 * A * C)
    y0 = (2 * A * E - B * D) / (B**2 - 4 * A * C)
    theta = 0.5 * np.arctan2(-B, (C - A)) * 180 / np.pi

    name = anno_path.name.rsplit("_", 1)[0] + ".png"
    if mode == "xyabt":
        return [name, x0, y0, a, b, theta]
    elif mode == "abcdef":
        return [name, A, B, C, D, E, F]


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
def write_an_obb(anno_path: str | Path) -> None:
    """
    - Parameters
        - anno_path: Path (path to an annotation image)
        - label_path: Path (path to the label directory)

    - Behavior

        Given a annotation image, this function write the parameters of the oriented bounding box that best fits the annotation to a text file using multi-threading.

        class: 0:head
        x1, y1: highest point
        x2, y2: rightmost point
        x3, y3: lowest point
        x4, y4: leftmost point

    - Returns
        - None
    """
    if isinstance(anno_path, str):
        anno_path = Path(anno_path)
    anno = Image.open(anno_path)
    data_class = anno_path.parent.name
    pixel_array = np.array(anno.getdata()).reshape(anno.size[1], anno.size[0])
    coords = np.array(np.where(pixel_array))[::-1]
    x1, y1 = coords[:, np.argmin(coords[1])] / pixel_array.T.shape
    x2, y2 = coords[:, np.argmax(coords[0])] / pixel_array.T.shape
    x3, y3 = coords[:, np.argmax(coords[1])] / pixel_array.T.shape
    x4, y4 = coords[:, np.argmin(coords[0])] / pixel_array.T.shape
    name = anno_path.name.rsplit("_", 1)[0] + ".txt"
    with open(anno_path.parents[1] / "labels" / name, "w") as f:
        f.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}")


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
    anno_paths = data["annotation_path"].to_list()
    thread_map(write_an_obb, anno_paths)


# %%
def plot_sample_data(data: pl.DataFrame, name: str, *args, **kwargs) -> None:
    """
    - Parameters:
        - `data`: pl.DataFrame (dataframe containing the image aname, image path, annotation path, and the ellipse parameters, the pixel size and the head circumference)

    - Behavior:

        Plot some sample images with their corresponding ellipse annotation.

    - Returns:
        None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    axes = axes.flatten()
    data = data.sample(fraction=1, shuffle=True)
    for i, ax in enumerate(axes):
        image = Image.open(data["image_path"][i])
        ax.imshow(image, cmap="gray")
        ellipse = Ellipse(
            (data["x0"][i], data["y0"][i]),
            data["a"][i] * 2,
            data["b"][i] * 2,
            angle=data["theta"][i],
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(ellipse)
        ax.set_title(data["image_name"][i])
    fig.suptitle(f"Sample images in {name} with their corresponding ellipse annotation")
    plt.show()


# %%
def plot_data(data: pl.DataFrame, *args, **kwargs) -> None:
    name = argname("data").split("_")[1]
    plot_sample_data(data, name)


if __name__ == "__main__":
    # %% [markdown]
    # # $\text{Reads and Analyze data}$

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

    # %%
    plot_data(data_train)

    # %%
    plot_data(data_val)

    # %%
    plot_data(data_test)

    # %% [markdown]
    # # $\text{Make OBB labels}$

    # %%
    write_oob_labels(data_train, root_dir)
    write_oob_labels(data_val, root_dir)
    write_oob_labels(data_test, root_dir)
