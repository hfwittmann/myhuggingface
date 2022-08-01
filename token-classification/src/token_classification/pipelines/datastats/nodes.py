"""
This is a boilerplate pipeline 'datastats'
generated using Kedro 0.18.2
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def save_histogram(k, file, title):
    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Plot the histogram
    plt.hist(k)
    plt.title(title)
    plt.xlabel("Number of words")
    plt.ylabel("Occurences")

    # Save the histogram
    plt.savefig(file)
    plt.close()

    return None


def calc_stats(dataset, datasetname):
    limit = 1000000

    Path(f"data/01_raw/{datasetname}/stats").mkdir(parents=True, exist_ok=True)

    D_stats = pd.DataFrame()

    for split, data in dataset.items():
        data = pd.DataFrame(data[:limit])
        lens = data["tokens"].apply(len)
        lens_ner = data["ner_tags"].apply(len)

        assert (lens == lens_ner).all()
        D_new = pd.DataFrame({"split": split, "lengths_mean": np.mean(lens), "lengths_std ": np.std(lens)}, index=[0])

        D_stats = pd.concat([D_stats, D_new], axis=0, ignore_index=True)
        save_histogram(
            lens,
            f"data/01_raw/{datasetname}/stats/hist_{split}.png",
            title=f"Number of words distribution in {datasetname} {split}",
        )

    D_stats.to_json(f"data/01_raw/{datasetname}/stats/mystats.json")

    return None
