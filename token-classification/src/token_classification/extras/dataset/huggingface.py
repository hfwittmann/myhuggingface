from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec

from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


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
        pass

    def _save(self, dataset: DatasetDict) -> None:
        """Saves data to the specified filepath."""

        save_path = get_filepath_str(self._filepath, self._protocol)

        # assume that we have already loaded the dataset called "dataset"
        for split, data in dataset.items():

            if self._subtype is None:
                data.save_to_disk(save_path)

            if self._subtype == "csv":
                data.to_csv(save_path, index=None)

            if self._subtype == "json":
                data.to_json(save_path)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
