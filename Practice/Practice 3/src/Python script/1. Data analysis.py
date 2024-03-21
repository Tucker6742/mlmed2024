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
from tqdm.contrib.concurrent import thread_map
from colour import Color
import warnings

# %%
warnings.filterwarnings("ignore")

# %% [markdown]
# # $\text{Utils functions}$


# %%
def convert_RGBA(image: Image.Image, alpha: float, color: Color) -> Image.Image:
    """
    - Parameters
        - image: Image.Image (Image from PIL)
        - alpha: float (Alpha value for the image)
        - color: list[int] (Color to be used for the image)

    -Behavior
        - Converts the image from B&W to RGBA

    - Returns
        - rgba_image: np.array (Image in RGBA format)
    """
    pixel_array = np.array(image)
    pixel_array_r = np.where(pixel_array > 0, color.rgb[0], 0)
    pixel_array_g = np.where(pixel_array > 0, color.rgb[1], 0)
    pixel_array_b = np.where(pixel_array > 0, color.rgb[2], 0)
    pixel_array_a = np.where(pixel_array > 0, alpha, 0)
    rgba_image = np.dstack((pixel_array_r, pixel_array_g, pixel_array_b, pixel_array_a))
    return rgba_image


# %%
def getting_df(data_path: Path, root_dir: Path) -> pl.DataFrame:
    """
    - Parameters:
        - data_path: Path to the data directory
        - root_dir: Path to the root directory

    - Behavior:
        - This function takes in the data split folder and and return the dataframe of the data

    - Returns:
        - df: `pl.DataFrame` (Dataframe consist of file names, image paths, infection mask paths, lung mask paths sorted by file names)
    """
    images_path = data_path / "images"
    infections_path = data_path / "infection masks"
    lungs_path = data_path / "lung masks"
    file_names = sorted(
        [f.stem for f in Path(images_path).glob("*.png")],
        key=lambda x: int(x.removesuffix(".png").split("_")[-1]),
    )
    image_paths = [str(images_path / f) + ".png" for f in file_names]
    infection_paths = [str(infections_path / f) + ".png" for f in file_names]
    lung_paths = [str(lungs_path / f) + ".png" for f in file_names]
    df = pl.DataFrame(
        {
            "file_name": file_names,
            "image_path": image_paths,
            "infection_path": infection_paths,
            "lung_path": lung_paths,
        }
    )
    return df


# %%
def plot_data(df: pl.DataFrame) -> None:
    """
    - Parameters:
        - df: `pl.DataFrame` (Dataframe consist of file names, image paths, infection mask paths, lung mask paths)

    - Behavior:
        - This function takes in the dataframe and plot 4 sample images, infection masks and lung masks and combine image(base image, lung mask in green and infection mask in red with black pixel is transparent)

    - Returns:
        - None
    """
    for i in range(4):
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        name = df["file_name"][i]
        image = Image.open(df["image_path"][i])
        infection_mask = Image.open(df["infection_path"][i])
        lung_mask = Image.open(df["lung_path"][i])
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[1].imshow(lung_mask, cmap="gray")
        ax[1].set_title("Lung Mask")
        ax[2].imshow(infection_mask, cmap="gray")
        ax[2].set_title("Infection Mask")
        lung_image = convert_RGBA(lung_mask, 0.6, Color("lime"))
        infection_image = convert_RGBA(infection_mask, 0.3, Color("brown"))
        ax[3].imshow(image, cmap="gray")
        ax[3].imshow(lung_image)
        ax[3].imshow(infection_image)
        ax[3].set_title("Combined image")
        fig.suptitle(f"Image {name}", fontsize=20)
        plt.show()


# %%
def getting_percentage(*dfs) -> None:
    """
    - Parameters:
        - dfs: `pl.DataFrame` (Dataframe consist of file names, image paths, infection mask paths, lung mask paths)

    - Behavior:
        - This function takes in the dataframes and print a percentage data frame which includes the name of the data split, number of image in the split, and percentage of the data split among all data

    - Returns:
        - None
    """
    total = sum([len(df) for df in dfs])
    names = argname("*dfs")
    percentage_df = pl.DataFrame(
        {
            "Data Split": names,
            "Number of Images": [len(df) for df in dfs],
            "Percentage": [np.round((len(df) / total) * 100, 2) for df in dfs],
        }
    )
    print(percentage_df)


# %%
def image_analysis(image_path: Path) -> dict:
    """
    - Parameters:
        - image_path: Path to the image

    - Behavior:
        - This function takes in the image path and print the dictionary of the image analysis which includes the image name, image size, and the number of channels in the image

    - Returns:
        - image_analysis: dict (Dictionary of the image analysis)
    """
    image_path = Path(image_path)
    image = Image.open(image_path)
    plt.imshow(image, cmap="gray")
    image_analysis = {
        "Image Name": image_path.stem,
        "Image Size": image.size,
        "Number of Channels": len(image.split()),
    }
    print(image_analysis)


# %%
def cal_IoU(infection_mask: Image.Image, lung_mask: Image.Image) -> float:
    """
    - Parameters:
        - infection_mask: Image.Image (Infection mask)
        - lung_mask: Image.Image (Lung mask)

    - Behavior:
        - This function takes in the infection mask and lung mask and calculate the intersection over union (IoU) of the masks

    - Returns:
        - iou: float (Intersection over union of the masks)
    """
    intersection = np.logical_and(infection_mask, lung_mask)
    union = np.logical_or(infection_mask, lung_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou


# %%
def ratio_white_pixels(image: Image.Image) -> float:
    """
    - Parameters:
        - image: Image.Image (Image)

    - Behavior:
        - This function takes in the image and calculate the percentage of white pixels in the image

    - Returns:
        - percentage: float (Percentage of white pixels in the image)
    """
    pixel_array = np.array(image)
    white_pixels = np.sum(pixel_array > 0)
    total_pixels = np.prod(pixel_array.shape)
    ratio = white_pixels / total_pixels
    return ratio


# %%
def get_linear_bounding_scatter(x: np.array, y: np.array, split: int) -> np.array:
    """
    - Parameters:
        - x: np.array (X values)
        - y: np.array (Y values)
        - split: int (Number of point to fit the line)

    - Behavior:
        - This function takes in the x and y values and calculate the upper, lower, left, right linear bounding of the scatter plot

    - Returns:
        - upper: np.array (Upper linear bounding)
        - lower: np.array (Lower linear bounding)
        - left: np.array (Left linear bounding)
        - right: np.array (Right linear bounding)
    """
    x_range = np.linspace(x.min(), x.max(), split)
    y_range = np.linspace(y.min(), y.max(), split)

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []

    for i, _ in enumerate(x_range[:-1]):
        index_x = np.where((x >= x_range[i]) & (x <= x_range[i + 1]))
        y_max = y[index_x].max()
        y_min = y[index_x].min()
        y_mins.append(y_min)
        y_maxs.append(y_max)

    for i, _ in enumerate(y_range[:-1]):
        index_y = np.where((y >= y_range[i]) & (y <= y_range[i + 1]))
        x_max = x[index_y].max()
        x_min = x[index_y].min()
        x_mins.append(x_min)
        x_maxs.append(x_max)

    xs = (x_range[:-1] + x_range[1:]) / 2
    ys = (y_range[:-1] + y_range[1:]) / 2

    upper = np.polyfit(xs, y_maxs, 1)
    lower = np.polyfit(xs, y_mins, 1)

    left = np.polyfit(x_mins, ys, 1)
    right = np.polyfit(x_maxs, ys, 1)

    return upper, lower, left, right


# %%
def process_one_row(row: dict[str, str]) -> pl.DataFrame:
    """
    - Parameters:
        - row: dict[str, str] (row dict of the dataframe)

    - Behavior:
        - This function takes in the row dict of the dataframe and return the list of the processed row that include the file name, height, width, Infection Mask Mask IoU, Infection Mask White Pixels %, Lung Mask White White Pixels %

    - Returns:
        - processed_df: pl.DataFrame (Dataframe of the processed row)
    """
    file_name = row["file_name"]
    image = Image.open(row["image_path"])
    infection_mask = Image.open(row["infection_path"])
    lung_mask = Image.open(row["lung_path"])
    height, width = image.size
    infection_mask = np.array(infection_mask)
    lung_mask = np.array(lung_mask)
    infection_mask_iou = cal_IoU(infection_mask, lung_mask)
    infection_mask_white_pixels = ratio_white_pixels(infection_mask)
    lung_mask_white_pixels = ratio_white_pixels(lung_mask)
    return [
        file_name,
        height,
        width,
        infection_mask_iou,
        infection_mask_white_pixels,
        lung_mask_white_pixels,
    ]


# %%
def dataframe_analysis(df: pl.DataFrame) -> None:
    """
    - Parameters:
        - df: `pl.DataFrame` (Dataframe consist of file names, image paths, infection mask paths, lung mask paths)

    - Behavior:
        - This function takes in the dataframe and create an dataframe which includes the name of the image, image height, width, mask IoU, and percentage of white pixels of each mask of each image and percentage of pixel is in infection but not in lung. After that it print the polars describe of the dataframe and make appropriate histogram and scatter plot

    - Returns:
        - analysis_df: `pl.DataFrame` (Dataframe consist of name of the image, image height, width, mask IoU, and percentage of white pixels of each mask of each image)
    """
    name = argname("df")
    print(f"Processing {name}")
    analysis_data = thread_map(process_one_row, df.to_dicts())
    analysis_df = pl.DataFrame(
        analysis_data,
        schema=[
            "file name",
            "height",
            "width",
            "Infection Mask IoU",
            "Infection Mask White Pixels %",
            "Lung Mask White Pixels %",
        ],
    )

    print(analysis_df.describe())

    # Infection mask IoU and White Pixels % Histogram between Infection and Lung Mask
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    ax[0].hist(analysis_df["Infection Mask IoU"], bins=20)
    ax[0].set_title("Infection Mask IoU Histogram")
    ax[0].set_xlabel("IoU")
    ax[0].set_ylabel("Frequency")

    ax[1].scatter(
        analysis_df["Infection Mask White Pixels %"],
        analysis_df["Lung Mask White Pixels %"],
    )
    upper, _, _, right = get_linear_bounding_scatter(
        analysis_df["Infection Mask White Pixels %"].to_numpy(),
        analysis_df["Lung Mask White Pixels %"].to_numpy(),
        split=10,
    )
    xs = np.linspace(0, 1, 100)
    y_upper = np.polyval(upper, xs)
    y_right = np.polyval(right, xs)
    ax[1].plot(
        xs,
        y_upper,
        color="red",
        label=f"Upper Bound y = {upper[0]:.2f}x {'+' if upper[1] > 0 else '-'} {abs(upper[1]):.2f}",
    )
    ax[1].plot(
        xs,
        y_right,
        color="blue",
        label=f"Right Bound y = {right[0]:.2f}x {'+' if right[1] > 0 else '-'} {abs(right[1]):.2f}",
    )
    ax[1].set_title("Infection Mask vs Lung Mask White Pixels %")
    ax[1].set_xlabel("Infection Mask White Pixels %")
    ax[1].set_ylabel("Lung Mask White Pixels %")
    ax[1].set_xlim([0, 0.7])
    ax[1].set_ylim([0, 0.7])
    ax[1].legend()
    plt.show()

    # Infection mask and Lung mask White Pixels % Histogram
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax[0].hist(analysis_df["Infection Mask White Pixels %"], bins=20)
    ax[0].set_title("Infection Mask White Pixels % Histogram")
    ax[0].set_xlabel("White Pixels %")
    ax[0].set_ylabel("Frequency")
    ax[1].hist(analysis_df["Lung Mask White Pixels %"], bins=20)
    ax[1].set_title("Lung Mask White Pixels % Histogram")
    ax[1].set_xlabel("White Pixels %")
    ax[1].set_ylabel("Frequency")
    plt.show()

    # Infection mask IoU vs White Pixels % of Infection and Lung Mask Scatter Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax[0].scatter(
        analysis_df["Infection Mask IoU"],
        analysis_df["Infection Mask White Pixels %"],
    )
    upper, lower, _, _ = get_linear_bounding_scatter(
        analysis_df["Infection Mask IoU"].to_numpy(),
        analysis_df["Infection Mask White Pixels %"].to_numpy(),
        split=10,
    )
    xs = np.linspace(0, 1, 100)
    y_upper = np.polyval(upper, xs)
    y_lower = np.polyval(lower, xs)
    ax[0].plot(
        xs,
        y_upper,
        color="red",
        label=f"Upper Bound y = {upper[0]:.2f}x {'+' if upper[1] > 0 else '-'} {abs(upper[1]):.2f}",
    )
    ax[0].plot(
        xs,
        y_lower,
        color="blue",
        label=f"Lower Bound y = {lower[0]:.2f}x {'+' if lower[1] > 0 else '-'} {abs(lower[1]):.2f}",
    )
    ax[0].legend()
    ax[0].set_title("Infection Mask IoU vs Infection Mask White Pixels %")
    ax[0].set_xlabel("IoU")
    ax[0].set_ylabel("Infection Mask White Pixels %")
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])

    ax[1].scatter(
        analysis_df["Infection Mask IoU"],
        analysis_df["Lung Mask White Pixels %"],
    )
    upper, lower, _, _ = get_linear_bounding_scatter(
        analysis_df["Infection Mask IoU"].to_numpy(),
        analysis_df["Lung Mask White Pixels %"].to_numpy(),
        split=10,
    )
    xs = np.linspace(0, 1, 100)
    y_upper = np.polyval(upper, xs)
    y_lower = np.polyval(lower, xs)
    ax[1].plot(
        xs,
        y_upper,
        color="red",
        label=f"Upper Bound y = {upper[0]:.2f}x {'+' if upper[1] > 0 else '-'} {abs(upper[1]):.2f}",
    )
    ax[1].plot(
        xs,
        y_lower,
        color="blue",
        label=f"Lower Bound y = {lower[0]:.2f}x {'+' if lower[1] > 0 else '-'} {abs(lower[1]):.2f}",
    )
    ax[1].legend()
    ax[1].set_title("Infection Mask IoU vs Lung Mask White Pixels %")
    ax[1].set_xlabel("IoU")
    ax[1].set_ylabel("Lung Mask White Pixels %")
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    plt.show()

    return analysis_df


if __name__ == "__main__":
    # %% [markdown]
    # # $\text{Read data}$

    # %%
    root_dir = Path("../../data/Infection Segmentation Data")

    # %%
    data_path_train: pathlib.Path = Path(f"{root_dir}/Train/COVID-19")
    data_train: pl.DataFrame = getting_df(data_path_train, root_dir)
    print(f"{data_train.shape = }")
    print(data_train.head())

    # %%
    data_path_val: pathlib.Path = Path(f"{root_dir}/Val/COVID-19")
    data_val: pl.DataFrame = getting_df(data_path_val, root_dir)
    print(f"{data_val.shape = }\n")
    print(data_val.head())

    # %%
    data_path_test: pathlib.Path = Path(f"{root_dir}/Test/COVID-19")
    data_test: pl.DataFrame = getting_df(data_path_test, root_dir)
    print(f"{data_test.shape = }\n")
    print(data_test.head())

    # %% [markdown]
    # # $\text{Plot data}$

    # %%
    plot_data(data_train)

    # %%
    plot_data(data_val)

    # %%
    plot_data(data_test)

    # %% [markdown]
    # # $\text{Data analysis}$

    # %%
    getting_percentage(data_train, data_val, data_test)

    # %%
    image_analysis(data_train["image_path"][0])

    # %%
    train_analysis = dataframe_analysis(data_train)

    # %%
    val_analysis = dataframe_analysis(data_val)

    # %%
    test_analysis = dataframe_analysis(data_test)
