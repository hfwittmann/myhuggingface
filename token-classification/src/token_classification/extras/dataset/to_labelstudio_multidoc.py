import json
from datasets import Dataset

from h11 import Data
from .to_labelstudio_onedoc import convert_one_document
from tqdm import tqdm
from datasets import dataset_dict


def convert_multiple_docs(md):

    limit = min(2000, len(md))

    md = md[:limit]
    md = Dataset.from_dict(md)

    return [convert_one_document(d) for ix, d in tqdm(enumerate(md), total=limit)]


if __name__ == "__main__":

    with open("token-classification/src/token_classification/extras/dataset/dataset_test.json", "r") as f:
        md = []
        for line in f:
            json_line = json.loads(line)
            md.append(json_line)

    d_converted = convert_multiple_docs(md)

    with open("token-classification/src/token_classification/extras/dataset/dataset_labelstudio_test.json", "w") as f:
        f.write("[\n")
        for ix, d in enumerate(d_converted):
            json.dump(d, f)
            # dont use after last element
            if ix < len(d_converted) - 1:
                f.write("\n,")

        f.write("\n]")

    print("finished")
