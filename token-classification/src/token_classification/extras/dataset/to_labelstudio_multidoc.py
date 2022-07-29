import json
from to_labelstudio_onedoc import convert_one_document


def convert_multiple_docs(md):

    limit = 100000
    return [convert_one_document(d) for ix, d in enumerate(md) if ix < limit]


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
