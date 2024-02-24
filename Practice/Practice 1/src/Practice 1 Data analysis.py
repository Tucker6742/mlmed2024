
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import polars.selectors as cs
from varname import argname

pl.Config.set_tbl_hide_dataframe_shape(True)


def formatting_data(data_path: str) -> pl.DataFrame:
    name = argname("data_path")
    print(f"\n\nReading {name}\n")
    raw_data: pl.DataFrame = pl.read_csv(data_path)
    class_col = raw_data.columns[-1]
    data = raw_data.select(
        value=pl.concat_list(pl.exclude(class_col)),
        signal_class=pl.col(class_col)
    ).cast({"signal_class": pl.Int32})
    print(
        data.group_by("signal_class")
        .agg(pl.count("value"))
        .sort("signal_class")
        .with_columns((pl.col("value") / pl.sum("value")).alias("percentage") * 100)
        .rename({"value": "count"})
    )
    return data


def plot_data(data: pl.DataFrame, name_dict: dict[int, str]) -> None:
    title = argname("data")
    print(f"\n\nPlotting {title}\n")
    data_group = data.group_by("signal_class").all().sort("signal_class")
    num_class = data_group.shape[0]
    scale = 10
    fig, ax = plt.subplots(
        num_class, 1,
        figsize=(
            scale * 3, num_class * scale
        ),
        layout="constrained"

    )
    for index_axis, axis in enumerate(ax):
        for signal in data_group["value"][index_axis]:
            axis.plot(signal)
        axis.set_title(f"Class {name_dict[index_axis]}", fontsize=30)
    fig.suptitle(f"Signal Classes {title}", fontsize=40)
    fig.savefig(f"../Figure/Signal Classes {title}")
    print(f"Figure saved as Signal Classes {title}.png")


def main():
    name_dict = {
        0: "N",
        1: "S",
        2: "V",
        3: "F",
        4: "Q"
    }
    data_path_train = "../archive/mitbih_train.csv"
    data_train = formatting_data(data_path_train)
    data_path_test = "../archive/mitbih_test.csv"
    data_test = formatting_data(data_path_test)
    plot_data(data_train, name_dict)
    plot_data(data_test, name_dict)


if __name__ == "__main__":
    main()
