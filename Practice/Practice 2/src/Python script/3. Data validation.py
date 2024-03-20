# %% [markdown]
# # $\text{Import library}$

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy.linalg as npla
from matplotlib.patches import Ellipse
from tqdm.contrib.concurrent import thread_map
from matplotlib.patches import Polygon

# %% [markdown]
# # $\text{Utils functions}$


# %%
def read_label_file(label_file: Path):
    """
    - Parameters:
        - label_file: Path to the label file

    - Behavior:
        - Reads the label file and returns the name and the coordinates of the bounding box

    - Returns:
        - A list with the name and the coordinates of the bounding box
    """
    name = label_file.stem
    with open(label_file, "r") as f:
        line = f.readlines()[0].split(" ")
        line = [name] + [float(x) for x in line]
    return line


# %%
def get_ellipse_param(anno_path: Path | str, mode="xyabt") -> list[str | float]:
    """
    - Parameters
        - anno_path: Path (path to the annotation image)

    - Behavior
        - Given an annotation image, this function returns the parameters of the ellipse that best fits the annotation.

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
def get_obb_point(
    x: float,
    y: float,
    a: float,
    b: float,
    t: float,
    image_path: Path | str,
    mode="xyxyxyxy",
) -> list[str | float]:
    """
    - Parameters
        - x: float (x-coordinate of the center of the ellipse)
        - y: float (y-coordinate of the center of the ellipse)
        - a: float (width of the ellipse)
        - b: float (height of the ellipse)
        - t: float (angle of the ellipse)
        - image_path: Path (path to the annotation image)
        - mode: str (mode of the output)

    - Behavior
        - Given an annotation image, this function returns the parameters of the oriented bounding box that best fits the annotation.

        - A: rightmost point after rotation
        - B: top point after rotation
        - C: leftmost point after rotation
        - D: bottom point after rotation
        - X: center of the ellipse

        - O1: bbox point between A, B
        - O2: bbox point between B, C
        - O3: bbox point between C, D
        - O4: bbox point between D, A

        - OO1 = OA + AO1 = OA + XB
        - OO2 = OB + BO2 = OB + XC
        - OO3 = OC + CO3 = OC + XD
        - OO4 = OD + DO4 = OD + XA

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
def get_df(label_path: Path) -> pl.DataFrame:
    """
    - Parameters:
        - label_path: Path (Path of the label directory)

    - Behavior
        - Read the each label file and convert it to a DataFrame
        - Concatenate all the DataFrames and return it

    - Return
        - df: pl.DataFrame (DataFrame of the label files)
    """

    label_files = list(label_path.glob("*.txt"))

    data = thread_map(read_label_file, label_files)
    bbox_df = pl.DataFrame(
        data,
        schema={
            "image_name": pl.Utf8,
            "class": pl.Int64,
            "x1": pl.Float64,
            "y1": pl.Float64,
            "x2": pl.Float64,
            "y2": pl.Float64,
            "x3": pl.Float64,
            "y3": pl.Float64,
            "x4": pl.Float64,
            "y4": pl.Float64,
        },
    )

    return bbox_df


# %%
def plots(image_name: str, data_split_type: str, root_dir: Path):
    """
    - Parameters:
        - image_name: str (Name of the image)
        - data_split_type: str (Type of the data split train/val/test)
        - root_dir: Path (Root directory of the dataset)

    - Behavior:
        - Read the image and the label file
        - get the ellipse parameters and the oriented bounding box parameters
        - plot the image, ellipse and the oriented bounding box in the same figure

    - Return:
        - None
    """

    image_path = root_dir / data_split_type / "images" / f"{image_name}.png"
    anno_path = (
        root_dir / data_split_type / "annotations" / f"{image_name}_Annotation.png"
    )

    ellipse_param = get_ellipse_param(anno_path=anno_path, mode="xyabt")

    bbox_param = get_obb_point(
        x=ellipse_param[1],
        y=ellipse_param[2],
        a=ellipse_param[3],
        b=ellipse_param[4],
        t=ellipse_param[5],
        image_path=image_path,
        mode="xyxyxyxy",
    )

    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ellipse = Ellipse(
        (ellipse_param[1], ellipse_param[2]),
        ellipse_param[3] * 2,
        ellipse_param[4] * 2,
        angle=ellipse_param[5],
        fill=False,
        color="r",
    )
    ax.add_patch(ellipse)
    polygon = Polygon(
        [
            (bbox_param[0] * image.size[0], bbox_param[1] * image.size[1]),
            (bbox_param[2] * image.size[0], bbox_param[3] * image.size[1]),
            (bbox_param[4] * image.size[0], bbox_param[5] * image.size[1]),
            (bbox_param[6] * image.size[0], bbox_param[7] * image.size[1]),
        ],
        closed=True,
        fill=False,
        color="g",
    )

    ax.add_patch(polygon)

    ax.set_title(f"{image_name}")

    plt.show()


if __name__ == "__main__":
    # %% [markdown]
    # # $\text{Setup parameters}$

    # %%
    split = "train"
    index = 0
    root_dir = Path("../../data")

    # %%
    split_label_path = Path(f"../../data/{split}/labels")

    split_label_path = get_df(split_label_path)

    # %%
    error_image = split_label_path.with_columns(
        has_negative_value=pl.min_horizontal(
            "x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"
        )
        < 0,
        has_over_one_value=pl.max_horizontal(
            "x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"
        )
        > 1,
    ).filter(pl.col("has_negative_value") | pl.col("has_over_one_value"))

    # %%
    plots(error_image["image_name"][index], split, root_dir)
