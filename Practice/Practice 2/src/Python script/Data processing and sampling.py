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
from shutil import copyfile, rmtree, move

# %% [markdown]
# # $\text{Utils functions}$


# %%
def sampling_index(dataframe: pl.DataFrame, config) -> list[pl.DataFrame]:
    """
    -Parameters
        dataframe: pl.DataFrame
            - The dataframe to be used for sampling
        config: dict
            - The configuration dictionary

    - Behaviour
        - This function will return the indices of the dataframe to be used for sampling of train, val, set

    -Returns
        - The indices of the dataframe to be used for sampling of train, val, set
    """

    dataframe = dataframe.sample(fraction=1, shuffle=True)
    train_size = int(dataframe.height * config.train_split)
    val_size = int(dataframe.height * config.val_split)
    test_size = dataframe.height - train_size - val_size
    train_df = dataframe[0:train_size]
    val_df = dataframe[train_size : train_size + val_size]
    test_df = dataframe[train_size + val_size :]
    return train_df, val_df, test_df


# %%
def create_directory_tree(root_path: Path):
    """
    -Parameters
        root_path: Path
            - The current path

    - Behaviour
        - This function will create the directory tree for the current path

    -Returns
        - None
    """
    images = root_path / "images"
    if images.exists():
        rmtree(images)
    images.mkdir(exist_ok=True)
    annotations = root_path / "annotations"
    if annotations.exists():
        rmtree(annotations)
    annotations.mkdir(exist_ok=True)
    labels = root_path / "labels"
    if labels.exists():
        rmtree(labels)
    labels.mkdir(exist_ok=True)
    for path in [images, annotations, labels]:
        for sub_path in ["train", "val", "test"]:
            (path / sub_path).mkdir(exist_ok=True)

    print(f"Directory tree created at {root_path}")


# %%
def copy(src_dir: Path, dest_dir: Path):
    """
    -Parameters
        src_dir: Path
            - The source directory
        dest_dir: Path
            - The destination directory

    - Behaviour
        - This function will move the file from the source directory to the destination directory

    -Returns
        - None
    """
    src = src_dir
    dest = dest_dir
    copyfile(src, dest)
    # print(f"Moved {file_name} from {src_dir} to {dest_dir}")


# %%
def move_files(*dfs, root_dir) -> None:
    """
    - Parameters
        dfs: pl.DataFrame
            - The dataframes that contain file to be moved

    - Behaviour
        - This function will move the files from the source directory to the destination directory

    - Returns
        - None
    """
    df_names = [df.split("_")[0] for df in argname("dfs")]

    for df, df_name in zip(dfs, df_names):
        print(f"Copying files for {df_name}\n")
        img_dirs = df["img_path"].to_list()
        img_dest = [root_dir / "images" / df_name / file.name for file in img_dirs]
        ann_dirs = df["anno_path"].to_list()
        ann_dest = [root_dir / "annotations" / df_name / file.name for file in ann_dirs]

        print(f"Copying {len(img_dirs)} images\n")
        thread_map(copy, img_dirs, img_dest)

        print(f"Copying {len(ann_dirs)} annotations\n\n")
        thread_map(copy, ann_dirs, ann_dest)

    print("Files moved successfully")
    move(root_dir / "test_set", root_dir / "images" / "evaluate_set")
    rmtree(root_dir / "training_set")

    print("Finished moving files")


# %% [markdown]
# # $\text{Config}$


# %%
class Config:
    def __init__(self):
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1


# %%
config = Config()

# %%
root_dir = Path("../../data")

# %% [markdown]
# # $\text{Building directory tree}$

if __name__ == "__main__":
    # %%
    create_directory_tree(root_dir)

    # %% [markdown]
    # # $\text{Splitting data directory}$

    # %%
    img_paths = sorted(Path(root_dir / "training_set").rglob("*HC.png"))
    anno_paths = sorted(Path(root_dir / "training_set").rglob("*Annotation.png"))

    # %%
    train_df, val_df, test_df = sampling_index(
        pl.DataFrame(
            {
                "img_path": img_paths,
                "anno_path": anno_paths,
            }
        ),
        config,
    )

    # %%
    move_files(train_df, val_df, test_df, root_dir=root_dir)

    # %%
