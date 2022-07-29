from __future__ import annotations
from pathlib import PurePosixPath
from posixpath import split
from typing import Any, Dict

import datasets
import fsspec
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class HuggingfaceDataSet(AbstractDataSet):
    """``HuggingfaceDataSet`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> HuggingfaceDataSet(filepath='/img/file/path')
    """

    def __init__(self, filepath: str, subtype: str, splits: list = ["train", "test", "validate"]):
        """Creates a new instance of HuggingfaceDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self._subtype = subtype
        self._splits = splits

    def _load(self):
        """Loads data from the image file.

        Returns:
            Data from  file
        """
        save_path = get_filepath_str(self._filepath, self._protocol)

        if self._subtype is None:
            split = "all"
            save_path_formatted = save_path.format(split=split)
            dataset = load_from_disk(save_path_formatted)

        if self._subtype is not None:
            datafiles = {split: save_path.format(split=split) for split in self._splits}
            dataset = load_dataset(self._subtype, data_files=datafiles)

        return dataset

    def _save(self, dataset: DatasetDict) -> None:
        """Saves data to the specified filepath."""

        save_path = get_filepath_str(self._filepath, self._protocol)

        if self._subtype is None:
            split = "all"
            save_path_formatted = save_path.format(split=split)
            dataset.save_to_disk(save_path_formatted)

        # assume that we have already loaded the dataset called "dataset"
        if self._subtype is not None:
            for split, data in dataset.items():
                save_path_formatted = save_path.format(split=split)

                if self._subtype == "csv":
                    data.to_csv(save_path_formatted, index=None)

                if self._subtype == "json":
                    data.to_json(save_path_formatted)

                if self._subtype == "labelstudio":
                    pass

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)


if __name__ == "__main__":

    for ix, d in enumerate(data):
        
        if ix > 3:
            break

        word_bounderies = np.cumsum([len(t) for t in d["tokens"]])
        
        
            myannotations = {"annotations": {
                "result": 
                "data": {
                    "text": " ".join(data[
                        "tokens"
                    ][
                        0
                    ])
                }
            }
        