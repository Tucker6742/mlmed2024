# %% [markdown]
# # $\text{Import libraries}$

# %% [markdown]
# ## $\text{Based libraries}$

# %%
import numpy as np
import matplotlib.pyplot as plt
from varname import argname
import polars as pl
from pathlib import Path
import subprocess
import shutil
import platform
import random

# %% [markdown]
# ## $\text{Pytorch libraries}$

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import torchmetrics.classification as tmc

# %% [markdown]
# ## $\text{Lightning libraries}$

# %%
import lightning.pytorch as ptl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner.tuning import Tuner

# %% [markdown]
# # $\text{Utils functions}$


# %%
def formatting_data(data_path: str) -> pl.DataFrame:
    """
    - Parameter
        data_path: Path (path to the data split csv file)

    - Behavior
        - Create the data frame from the csv file
        - Create and print the analysis of the data

    - Return
        data: pl.DataFrame (the data frame)
        data_percentage: pl.DataFrame (the analysis of the data)
    """
    name = argname("data_path")
    print(f"Reading {name}\n")
    raw_data = pl.read_csv(data_path)
    class_col = raw_data.columns[-1]
    data = (
        raw_data.select(
            value=pl.concat_list(pl.exclude(class_col)), signal_class=pl.col(class_col)
        )
        .cast({"signal_class": pl.Int32})
        .sort("signal_class")
    )
    data_percentage = (
        data.group_by("signal_class")
        .agg(pl.count("value"))
        .sort("signal_class")
        .with_columns((pl.col("value") / pl.sum("value")).alias("percentage") * 100)
        .rename({"value": "count"})
    )
    print(data_percentage)
    return data, data_percentage


if __name__ == "__main__":
    # %% [markdown]
    # # $\text{Config class}$

    # %%
    class Config:
        def __init__(self, weight):
            self.train_split = 0.7
            self.val_split = 0.2
            self.test_split = 0.1
            self.lr = 0.0001
            self.weight = weight
            self.batch_size = 256

    # %%
    torch.manual_seed(6742)
    random.seed(6742)

    # %% [markdown]
    # # $\text{Read and prepare data}$

    # %%
    path_train = Path("../../archive/mitbih_train.csv")
    path_test = "../../archive/mitbih_test.csv"

    # %%
    data_train, data_train_percentage = formatting_data(path_train)

    # %%
    cut_off_train = int(
        data_train_percentage[0, "count"]
        - np.ceil(data_train_percentage[1:, "count"].mean())
    )

    # %%
    data_train = data_train[cut_off_train:, :]

    # %%
    data_train_percentage = (
        data_train.group_by("signal_class")
        .agg(pl.count("value"))
        .sort("signal_class")
        .with_columns((pl.col("value") / pl.sum("value")).alias("percentage") * 100)
        .rename({"value": "count"})
    )
    print(data_train_percentage)

    # %%
    weights = 1 / data_train_percentage[:, -1].to_numpy()

    # %%
    CONFIG = Config(weights)

    # %%
    data_val, data_val_percentage = formatting_data(path_test)

    # %%
    cut_off_val = int(
        data_val_percentage[0, "count"]
        - np.ceil(data_val_percentage[1:, "count"].mean())
    )

    # %%
    data_val = data_val[cut_off_val:, :]

    data_val_percentage = (
        data_val.group_by("signal_class")
        .agg(pl.count("value"))
        .sort("signal_class")
        .with_columns((pl.col("value") / pl.sum("value")).alias("percentage") * 100)
        .rename({"value": "count"})
    )
    print(data_val_percentage)

    # %%
    data_train = data_train.sample(fraction=1, shuffle=True)
    n = int(data_train.shape[0] * 7 / 8)
    data_train, data_test = data_train.head(n), data_train.tail(-n)

    # %%
    data_test.rows

    # %%
    data_train.rows

    # %%
    data_val.rows

    # %%
    name_dict: dict[int, str] = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

    # %% [markdown]
    # # $\text{Make data config}$

    # %%
    class Signal_Dataset(Dataset):
        def __init__(self, data_df):
            self.data_df = data_df

        def __len__(self):
            return len(self.data_df)

        def __getitem__(self, idx):
            signal = torch.tensor(self.data_df["value"][idx]).unsqueeze(0)
            label = torch.tensor(self.data_df["signal_class"][idx])
            return signal, label

    # %%
    class Dataset_config:
        def __init__(self, data_df, config):
            if not isinstance(data_df, list):
                self.data_dataset = Signal_Dataset(data_df)
                self.weight = torch.from_numpy(config.weights).float()
                self.train_size = int(config.train_split * len(self.data_dataset))
                self.val_size = int(config.val_split * len(self.data_dataset))
                self.test_size = (
                    len(self.data_dataset) - self.train_size - self.val_size
                )

                self.train_dataset, self.val_dataset, self.test_dataset = (
                    torch.utils.data.random_split(
                        self.data_dataset,
                        [self.train_size, self.val_size, self.test_size],
                    )
                )
            else:
                self.train_dataset = Signal_Dataset(data_df[0])
                self.val_dataset = Signal_Dataset(data_df[1])
                self.test_dataset = Signal_Dataset(data_df[2])

            self.train_dataloader = DataLoader(
                self.train_dataset,
                shuffle=True,
                # num_workers=2,
                batch_size=config.batch_size,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                shuffle=False,
                # num_workers=2,
                batch_size=config.batch_size,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                shuffle=False,
                # num_workers=2,
                batch_size=config.batch_size,
            )

    # %%
    DATA_SIGNAL = Dataset_config([data_train, data_val, data_test], CONFIG)

    # %%
    DATA_SIGNAL.train_dataset.__len__(), DATA_SIGNAL.val_dataset.__len__(), DATA_SIGNAL.test_dataset.__len__()

    # %% [markdown]
    # # $\text{Make model}$

    # %% [markdown]
    # ## $\text{Base Pytorch model}$

    # %%
    class ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(32, 32, 5, padding="same")
            self.conv2 = nn.Conv1d(32, 32, 5, padding="same")
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(5, 2)

        def forward(self, signal):
            identity = signal
            signal = self.conv1(signal)
            signal = self.relu(signal)
            signal = self.conv2(signal)
            signal = identity + signal
            signal = self.relu(signal)
            signal = self.pool(signal)
            return signal

    class Base_Pytorch_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 5, padding="same")
            self.Res_series = nn.Sequential(*[ResBlock()] * 5)
            self.fc1 = nn.Linear(32, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 5)

        def forward(self, signal):
            signal = self.conv1(signal)
            signal = self.Res_series(signal)
            signal = signal.flatten(1)
            signal = nn.AdaptiveAvgPool1d(32)(signal)
            signal = self.fc1(signal)
            signal = self.relu(signal)
            signal = self.fc2(signal)
            return signal

    # %%
    ECG_model = Base_Pytorch_Model()

    # %%
    data_iter = iter(DATA_SIGNAL.train_dataloader)

    # %%
    summary(ECG_model, next(data_iter)[0].shape, depth=5)

    # %% [markdown]
    # ## $\text{Lightning model}$

    # %%
    class Lightning_Wrapper(ptl.LightningModule):
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.learning_rate = config.lr
            if hasattr(config, "weight"):
                self.loss_fn = nn.CrossEntropyLoss(torch.tensor(config.weight).float())
            else:
                self.loss_fn = nn.CrossEntropyLoss()
            self.save_hyperparameters("learning_rate")

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            #         optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
            return optimizer

        def forward(self, signal):
            return self.model(signal)

        # __________________TRAIN_____________________________________________________

        def training_step(self, batch, batch_idx):
            try:
                signal, label = batch
                result = self(signal)

                predicted_class = result.argmax(dim=-1).flatten()

                accuracy = predicted_class == label

                loss = self.loss_fn(result, label)

                logs = {
                    "train_acc": accuracy.count_nonzero().item() / len(accuracy) * 100,
                }
                self.log_dict(logs)
                self.log("train_loss", loss, prog_bar=True)
                return loss
            except:
                print("Something wrong train upper")
                print(f"{batch_idx = }")
                print(f"{label.shape = }")
                print(f"{label.type() = }")
                print(f"{label = }")
                print(f"{result.shape = }")
                print(f"{result.type() = }")
                print(f"{result.mean(0) = }")
                print(f"{predicted_class = }")
                print(f"{accuracy = }")
                print(f"{loss = }\n\n")

        # __________________VALIDATION_____________________________________________________

        def validation_step(self, batch, batch_idx):
            try:
                signal, label = batch
                result = self(signal)

                predicted_class = result.argmax(dim=-1).flatten()

                accuracy = predicted_class == label

                loss = self.loss_fn(result, label)

                logs = {
                    "val_acc": accuracy.count_nonzero().item() / len(accuracy) * 100,
                }
                self.log_dict(logs)
                self.log("val_loss", loss, prog_bar=True)
                return loss
            except:
                print("Something wrong val upper")
                print(f"{batch_idx = }")
                print(f"{label.shape = }")
                print(f"{label.type() = }")
                print(f"{label = }")
                print(f"{result.shape = }")
                print(f"{result.type() = }")
                print(f"{result.mean(0) = }")
                print(f"{predicted_class = }")
                print(f"{accuracy = }")
                print(f"{loss = }\n\n")

        # __________________TEST_____________________________________________________

        def test_step(self, batch, batch_idx):
            try:
                signal, label = batch
                result = self(signal)

                predicted_class = result.argmax(dim=-1).flatten()

                accuracy = predicted_class == label

                loss = self.loss_fn(result, label)

                logs = {
                    "test_acc": accuracy.count_nonzero().item() / len(accuracy) * 100,
                }
                self.log_dict(logs)
                self.log("test_loss", loss, prog_bar=True)
                return loss
            except:
                print("Something wrong test upper")
                print(f"{batch_idx = }")
                print(f"{label.shape = }")
                print(f"{label.type() = }")
                print(f"{label = }")
                print(f"{result.shape = }")
                print(f"{result.type() = }")
                print(f"{result.mean(0) = }")
                print(f"{predicted_class = }")
                print(f"{accuracy = }")
                print(f"{loss = }\n\n")

        # _____________________PREDICT_____________________________________________________

        def predict_step(self, batch, batch_idx):
            try:
                signal, label = batch
                result = self(signal)

                return (result, label)
            except:
                print("Something wrong test upper")
                print(f"{batch_idx = }")
                print(f"{label.shape = }")
                print(f"{label.type() = }")
                print(f"{label = }")
                print(f"{result.shape = }")
                print(f"{result.type() = }")
                print(f"{result.mean(0) = }")

    # %%
    # base_model.cpu()
    # lightning_model.cpu()
    # del trainer, base_model, lightning_model
    # del tuner

    import gc

    gc.collect()

    # %% [markdown]
    # # $\text{Logging}$

    # %%
    shutil.rmtree("lightning_logs", ignore_errors=True)
    Path("lightning_logs").mkdir(exist_ok=True)

    # %%
    try:
        if platform.system() == "Windows":
            subprocess.run(["kill", "-name" "tensorboard"])
        else:
            subprocess.run(["killall", "tensorboard"])
    except:
        pass

    # %%
    logger = TensorBoardLogger("lightning_logs", name="", version=0)
    checkpoint_callback_train = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        # mode="min",
        dirpath="lightning_logs/version_0/checkpoints/train/",
        filename="train_model-{epoch}-{train_loss}-{val_loss}",
        verbose=True,
        # every_n_train_steps = 100,
        save_on_train_epoch_end=True,
    )
    checkpoint_callback_val = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        # mode="min",
        dirpath="lightning_logs/version_0/checkpoints/val/",
        filename="val_model-{epoch}-{train_loss}-{val_loss}",
        verbose=True,
        # every_n_train_steps = 100,
        save_on_train_epoch_end=True,
    )

    trainer = ptl.Trainer(
        #     fast_dev_run = True,
        max_epochs=100,
        callbacks=[
            # utils_callbacks,
            checkpoint_callback_train,
            checkpoint_callback_val,
        ],
        #     profiler="simple",
        accelerator="auto",
        #     accumulate_grad_batches=64,
        benchmark=True,
        # check_val_every_n_epoch=1,
        #     gradient_clip_val = 0.5,
        logger=logger,
        log_every_n_steps=100,
    )

    base_model = Base_Pytorch_Model()
    lightning_model = Lightning_Wrapper(base_model, CONFIG)

    # %% [markdown]
    # # $\text{Trainning step}$

    # %%
    tuner = Tuner(trainer)
    tuner.lr_find(
        lightning_model,
        train_dataloaders=DATA_SIGNAL.train_dataloader,
        val_dataloaders=DATA_SIGNAL.val_dataloader,
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=DATA_SIGNAL.train_dataloader,
        val_dataloaders=DATA_SIGNAL.val_dataloader,
    )

    # %% [markdown]
    # # $\text{Testing step}$

    # %% [markdown]
    # ## $\text{Train model best}$

    # %%
    train_pth = list(Path("lightning_logs").rglob("train_model*.ckpt"))[0]
    print(train_pth)

    # %%
    train_result = trainer.test(
        lightning_model, dataloaders=DATA_SIGNAL.test_dataloader, ckpt_path=train_pth
    )

    # %% [markdown]
    # ## $\text{Val model best}$

    # %%
    val_pth = list(Path("lightning_logs").rglob("val_model*.ckpt"))[0]
    print(val_pth)

    # %%
    val_result = trainer.test(
        lightning_model, dataloaders=DATA_SIGNAL.test_dataloader, ckpt_path=val_pth
    )

    # %%
    if train_result[0]["test_loss"] < val_result[0]["test_loss"]:
        best_model_path = train_pth
        print("Train model is better")
        print(f"train_model loss = {train_result[0]['test_loss']}")
        print(f"train_model accuracy = {train_result[0]['test_acc']}")
    else:
        best_model_path = val_pth
        print("Val model is better")
        print(f"val_model loss = {val_result[0]['test_loss']}")
        print(f"val_model accuracy = {val_result[0]['test_acc']}")

    # %% [markdown]
    # # $\text{Predicting step}$

    # %%
    result = trainer.predict(
        lightning_model,
        dataloaders=DATA_SIGNAL.test_dataloader,
        ckpt_path=best_model_path,
    )

    # %%
    predict_output = nn.Softmax(dim=-1)(torch.cat([x[0] for x in result]))
    predict_truth = torch.cat([x[1] for x in result])
    predict_class = predict_output.argmax(dim=-1)
    num_class = len(torch.unique(predict_truth))

    # %% [markdown]
    # # $\text{Metrics calculation}$

    # %% [markdown]
    # ## $\text{Accuracy}$

    # %%
    accuracy = tmc.MulticlassAccuracy(num_classes=num_class, average="none")
    accuracy(predict_output, predict_truth)

    # %% [markdown]
    # ## $\text{Precision}$

    # %%
    precision = tmc.MulticlassPrecision(
        num_classes=num_class,
        average="none",
    )

    precision(predict_output, predict_truth)

    # %% [markdown]
    # ## $\text{Recall}$

    # %%
    recall = tmc.MulticlassRecall(
        num_classes=num_class,
        average="none",
    )

    recall(predict_output, predict_truth)

    # %% [markdown]
    # ## $\text{F1 score}$

    # %%
    F1_score = tmc.MulticlassF1Score(
        num_classes=num_class,
        average="none",
    )

    F1_score(predict_output, predict_truth)

    # %% [markdown]
    # ## $\text{ROC}$

    # %%
    ROC = tmc.MulticlassROC(
        num_classes=num_class,
    )

    ROC.update(predict_output, predict_truth)

    fig_ROC, ax_ROC = ROC.plot(score=True)

    handle_ROC, label_ROC = ax_ROC.get_legend_handles_labels()
    label_ROC = [f"{name_dict[int(x.split()[0])]} {x.split()[1]}" for x in label_ROC]
    ax_ROC.legend(handle_ROC, label_ROC)

    plt.show()

    # %% [markdown]
    # ## $\text{Precision Recall curve}$

    # %%
    PRC = tmc.MulticlassPrecisionRecallCurve(
        num_classes=num_class,
    )

    PRC.update(predict_output, predict_truth)

    fig_PRC, ax_PRC = PRC.plot(score=True)
    handles, previous_labels = ax_PRC.get_legend_handles_labels()

    previous_labels = [
        f"{name_dict[int(x.split()[0])]} {x.split()[1]}" for x in previous_labels
    ]
    print(previous_labels)
    ax_PRC.legend(handles, previous_labels)

    plt.show()

    # %% [markdown]
    # ## $\text{Confusion matrix}$

    # %%
    confusion_matrix = tmc.MulticlassConfusionMatrix(
        num_classes=num_class,
        normalize="true",
    )

    confusion_matrix.update(predict_class, predict_truth)

    fig_confusion_matrix, ax_confusion_matrix = confusion_matrix.plot(
        labels=name_dict.values()
    )
    plt.show()

    # %%
