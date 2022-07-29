import json
import numpy as np


# https://huggingface.co/datasets/conll2003
tagging_scheme = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}

tagging_scheme_inverted = {value: key.split("-")[-1] for key, value in tagging_scheme.items()}
# {0: "O", 1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC", 7: "MISC", 8: "MISC"}


def convert_one_document(d):

    word_bounderies = np.cumsum([0] + [len(t) + 1 for t in d["tokens"]])
    word_bounderies = [int(w) for w in word_bounderies]

    annotations = [
        {
            "result": [
                {
                    "value": {
                        "start": word_bounderies[ix],
                        "end": word_bounderies[ix + 1],
                        "labels": [tagging_scheme_inverted[n]],
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                }
                for ix, n in enumerate(d["ner_tags"])
                if n != 0
            ]
        }
    ]

    onedocument = {"annotations": annotations, "data": {"text": " ".join(d["tokens"])}, "meta": {}}

    return onedocument


if __name__ == "__main__":

    with open("token-classification/src/token_classification/extras/dataset/dataset_test.json", "r") as f:
        for line in f:
            d = json.loads(line)
            break

    d_converted = convert_one_document(d)

    print("finished")
