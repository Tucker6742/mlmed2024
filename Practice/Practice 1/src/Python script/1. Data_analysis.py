# %% [markdown]
# # $\text{Import library}$

# %%
import polars as pl
import matplotlib.pyplot as plt
from varname import argname
import pathlib
from pathlib import Path

# %% [markdown]
# # $\text{Utils functions}$


# %%
def formatting_data(data_path: Path) -> pl.DataFrame:
    """
    - Parameter
        data_path: Path (path to the data split csv file)

    - Behavior
        - Create the data frame from the csv file
        - Print the analysis of the data

    - Return
        data: pl.DataFrame (the data frame)
    """
    raw_data: pl.DataFrame = pl.read_csv(data_path)
    class_col: str = raw_data.columns[-1]
    data: pl.DataFrame = raw_data.select(
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


# %%
def plot_data(data: pl.DataFrame, name_dict: dict[int, str]) -> None:
    """
    - Parameter
        data: pl.DataFrame (the data frame)
        name_dict: dict[int, str] (the dictionary of the signal class)

    - Behavior
        - Plot the line chart of all signal of all classes and the pie chart of the percentage number of each signal classes
        - Save the figures in the Figure folder

    - Return
        None
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


if __name__ == "__main__":
    # %% [markdown]
    # # $\text{Reads and Analyze data}$

    # %%
    name_dict: dict[int, str] = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

    # %%
    data_path_train: pathlib.Path = Path("../../archive/mitbih_train.csv")
    data_train: pl.DataFrame = formatting_data(data_path_train)

    # %%
    data_path_test = Path("../../archive/mitbih_test.csv")
    data_test: pl.DataFrame = formatting_data(data_path_test)

    # %%
    plot_data(data_train, name_dict)

    # %%
    plot_data(data_test, name_dict)
