# %% [markdown]
# # $\text{Import libraries}$

# %%
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import shutil
import polars as pl
import numpy as np

# %% [markdown]
# ## $\text{Torch libraries}$

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% [markdown]
# # $\text{Utils functions}$


# %%
def ellipse_circumference(a: float, b: float, size: float) -> float:
    a = a * size / 2
    b = b * size / 2
    return np.pi * (3 * (a + b) - np.sqrt(10 * a * b + 3 * (a**2 + b**2)))


if __name__ == "__main__":
    # %% [markdown]
    # # $\text{Directories setup}$

    # %%
    data_root_dir = Path("/kaggle/input/fetal-head-circumference/data")
    evaluation_dir = Path(
        "/kaggle/input/fetal-head-circumference/data/images/evaluate_set"
    )
    test_dir = Path("/kaggle/input/fetal-head-circumference/data/test/images")

    # %%
    root_dir = Path("../../results")
    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # %% [markdown]
    # # $\text{Setup model}$

    # %%
    Path(root_dir / "work").mkdir(parents=True, exist_ok=True)

    # %%
    model = YOLO(root_dir / "work/yolov8n-obb.pt")

    # %%
    with open(root_dir / "work/yolo.yaml", "w") as f:
        f.write(
            f"""
        path: {data_root_dir} # dataset root dir
        train: train/images 
        val: val/images 
        test: test/images

        # Classes for DOTA 1.0
        names:
        0: head
        """
        )

    # %%
    with open(root_dir / "work/yolo_test.yaml", "w") as f:
        f.write(
            f"""
        path: {data_root_dir} # dataset root dir
        train: train/images 
        val: test/images

        # Classes for DOTA 1.0
        names:
        0: head
        """
        )

    # %% [markdown]
    # # $\text{Traning}$

    # %%
    train_results = model.train(data=root_dir / "work/yolo.yaml", epochs=200)

    # %% [markdown]
    # # $\text{Testing}$

    # %%
    best_path = list(Path(".").rglob("best.pt"))[0]

    # %%
    best_model = YOLO(best_path)

    # %%
    best_model.val(data=root_dir / "work/yolo_test.yaml")

    # %% [markdown]
    # # $\text{Metrics}$

    # %%
    Path(root_dir / "test").mkdir(parents=True, exist_ok=True)

    # %%
    pixel_df = pl.read_csv(
        data_root_dir / "training_set_pixel_size_and_HC.csv",
        new_columns=["file_names", "pixel_size", "head_circumference"],
    )

    # %% [markdown]
    # ## $\text{Circumference}$

    # %%
    test_results = best_model(test_dir)

    # %%
    if Path(root_dir / "test/test_result").exists():
        shutil.rmtree(root_dir / "test/test_result")

    # %%
    file_names = []
    ellipse_params = []
    Path(root_dir / "test/test_result").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(test_results):
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        name = r.path.rsplit("/", 1)[-1]
        file_names.append(name)
        ellipse_params.append(r.obb.xywhr[0].flatten().cpu().tolist())
        r.save(root_dir / "test/test_result/{name}")
        # Show results to screen (in supported environments)
        if i % 20 == 0:
            plt.imshow(im_rgb)
            plt.title(name)
            plt.show()

    # %%
    test_result_df = pl.DataFrame(
        [file_names, ellipse_params], schema=["file_names", "ellipse_params"]
    )

    # %%
    test_result_df = test_result_df.with_columns(
        pl.col("ellipse_params").list.to_struct()
    ).unnest("ellipse_params")
    test_result_df.columns = ["file_names", "|x", "y", "w", "h", "theta"]
    test_result_df = test_result_df.join(pixel_df, on="file_names")

    # %%
    w_s = test_result_df["w"].to_numpy()
    h_s = test_result_df["h"].to_numpy()
    sizes = test_result_df["pixel_size"].to_numpy()
    truth_circum = test_result_df["head_circumference"].to_numpy()

    # %%
    circumferences = np.array(
        [ellipse_circumference(w, h, size) for (w, h, size) in zip(w_s, h_s, sizes)]
    )

    # %%
    test_result_df = test_result_df.with_columns(predicted_circumference=circumferences)

    # %%
    rmse_loss = torch.sqrt(
        F.mse_loss(torch.tensor(truth_circum), torch.tensor(circumferences))
    )

    # %%
    print(f"RMSE loss for circumference is {rmse_loss.item()}")

    # %% [markdown]
    # # $\text{Evaluate}$

    # %%
    evaluate_results = best_model(evaluation_dir)

    # %%
    if Path(root_dir / "evaluate/evaluate_result").exists():
        shutil.rmtree(root_dir / "evaluate/evaluate_result")

    # %%
    evaluate_file_names = []
    evaluate_ellipse_params = []
    Path(root_dir / "evaluate/evaluate_result").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(evaluate_results):
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        name = r.path.rsplit("/", 1)[-1]
        evaluate_file_names.append(name)
        evaluate_ellipse_params.append(r.obb.xywhr[0].flatten().cpu().tolist())
        r.save(root_dir / "evaluate_result/{name}")
        # Show results to screen (in supported environments)
        if i % 20 == 0:
            plt.imshow(im_rgb)
            plt.title(name)
            plt.show()

    # %%
    evaluate_result_df = pl.DataFrame(
        [evaluate_file_names, evaluate_ellipse_params],
        schema=["filename", "ellipse_params"],
    )

    # %%
    evaluate_result_df = evaluate_result_df.with_columns(
        pl.col("ellipse_params").list.to_struct()
    ).unnest("ellipse_params")
    evaluate_result_df.columns = [
        "filename",
        "center_x_mm",
        "center_y_mm",
        "semi_axes_a_mm",
        "semi_axes_b_mm",
        "angle_rad",
    ]

    # %%
    evaluate_result_df

    # %%
    evaluate_result_df.write_csv("evaluate_resull.csv")
