import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import polars.selectors as cs
from varname import argname
import pathlib
from pathlib import Path

pl.Config.set_tbl_hide_dataframe_shape(True)


def formatting_data(data_path: Path) -> pl.DataFrame:
    """
    Input:
    - data_path: str

    Behavior:

        This function reads the data from the given path and returns a DataFrame with the following columns:
        - value: `List[float]`. The signal_class column is the numeric type of the signal (0-4).
        - signal_class: `int32`. The value column is the list of all the signal belong to corresponding signal class.

        The function also prints the count and percentage of each signal class.
    """
    name = argname("data_path")
    print(f"\n\nReading {name}\n")
    raw_data: pl.DataFrame = pl.read_csv(data_path)
    class_col = raw_data.columns[-1]
    data = raw_data.select(
        value=pl.concat_list(pl.exclude(class_col)), signal_class=pl.col(class_col)
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
    """
    Input:
    - data: pl.DataFrame
    - name_dict: dict[int, str]

    Behavior:

        This function plots the signals of each signal class in the given data.\n
        The function saves the figure as "Signal Classes {data}.png".\n

        The function plot the percentage of each signal class in the given data with pie chart.
        The function saves the figure as "Signal Classes {data}.png".


    """
    title = argname("data")

    data_group = data.group_by("signal_class")

    data_line = data_group.all().sort("signal_class").rename({"value": "count"})

    data_pie = (
        data_group.agg(pl.count("value").alias("count"))
        .sort("signal_class")
        .with_columns(
            percentage=(pl.col("count") / pl.sum("count")).alias("percentage") * 100
        )
    )

    num_class = data_line.shape[0]

    scale = 10

    fig_line, ax_line = plt.subplots(
        num_class, 1, figsize=(scale * 3, num_class * scale), layout="constrained"
    )

    print("Begin to plot line chart. Please wait for a while.")

    for index_axis, axis in enumerate(ax_line):

        for signal in data_line["count"][index_axis]:

            axis.plot(signal)

        axis.set_title(f"Class {name_dict[index_axis]}", fontsize=3 * scale)

    fig_line.suptitle(f"Signal Classes {title}", fontsize=4 * scale)

    fig_line.savefig(f"../../Figure/Signal Classes {title}")
    plt.show()

    print("Line chart has been saved.")

    # Pie chart

    fig_pie, ax_pie = plt.subplots(
        1, 1, figsize=(scale * 3, scale * 3), layout="constrained"
    )

    print("Begin to plot pie chart. Please wait for a while.")

    ax_pie.pie(
        data_pie["percentage"],
        labels=[name_dict[i] for i in data_pie["signal_class"]],
        autopct="%1.2f%%",
        startangle=90,
        textprops=dict(fontsize=3.5 * scale),
    )

    ax_pie.set_title(f"Signal Classes {title}", fontsize=5 * scale)

    fig_pie.savefig(f"../../Figure/Signal Classes {title} Pie Chart")
    plt.show()

    print("Pie chart has been saved.")


def main():
    name_dict = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
    data_path_train = Path("../archive/mitbih_train.csv")
    data_train = formatting_data(data_path_train)
    data_path_test = Path("../archive/mitbih_test.csv")
    data_test = formatting_data(data_path_test)
    plot_data(data_train, name_dict)
    plot_data(data_test, name_dict)


if __name__ == "__main__":
    main()
