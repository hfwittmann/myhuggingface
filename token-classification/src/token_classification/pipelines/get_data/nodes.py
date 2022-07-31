"""
This is a boilerplate pipeline 'get_data'
generated using Kedro 0.18.2
"""
from __future__ import annotations
from datasets import Dataset
from datasets import load_dataset


import json
import os
import re
from functools import partial
from glob import glob
from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict

from tokenizers import Encoding
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def get_data_from_web():

    dataset = load_dataset("conll2003")

    d_with_split_information = dict()

    for key in dataset:
        blub = pd.DataFrame(dataset[key])
        blub["split"] = key
        d_with_split_information[key] = Dataset.from_pandas(blub)

    dataset_out = DatasetDict(d_with_split_information)

    return dataset_out  # , dataset, dataset


def charix_to_tokenix(txt, charix):
    counter = -1
    space_counter_active = False
    # re.sub(r" {1,}", " ", txt)

    for i in range(charix):
        if re.search(r"\s", txt[i]) and not space_counter_active:
            # if txt[i] in [" ","\n"] and not space_counter_active:
            counter += 1
            space_counter_active = True
        elif re.search(r"\s", txt[i]):
            space_counter_active = True
        elif not re.search(r"\s", txt[i]):
            space_counter_active = False

    return counter


def align_tokens_and_annotations_bilou(tokens, char_to_token, annotations):
    # tokens = tokenized.tokens
    aligned_labels = ["O"] * len(
        tokens
    )  # Make a list to store our labels the same length as our tokens
    for anno in annotations:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = char_to_token(charix=char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)
        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            prefix = (
                "U"  # This annotation spans one token so is prefixed with U for unique
            )
            aligned_labels[token_ix] = f"{prefix}-{anno['label']}"

        else:

            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"
    return aligned_labels


def get_texts(annotations_json):

    manydoc_json = list()
    for ix, one_document in tqdm(
        enumerate(annotations_json), total=len(annotations_json), smoothing=0.1
    ):

        text_filename = one_document["text"].replace("/data", "data")
        myid = text_filename.split("/")[-1].split("-")[0]

        mypaths = glob(
            f"data/upload/1/*{myid}*", recursive=True
        )  # TODO : auslagern in parameters
        assert len(mypaths) == 1, "should have one and only one match"
        text_filename = mypaths[0]

        with open(text_filename, "r") as file:
            text = file.read()

        if "label" in one_document.keys():
            annotations = one_document["label"]

            for idx, annotation in enumerate(annotations):
                start = annotation["start"]
                end = annotation["end"]
                annotations[idx]["label"] = annotations[idx]["labels"][0]
                annotations[idx]["text"] = text[start:end]

            # tokenized_batch: BatchEncoding = tokenizer(text)
            # tokenized_text: Encoding = tokenized_batch[0]

            # tokens = tokenized_text.tokens
            tokens = text.split()

            charix_to_tokenix_txt = partial(charix_to_tokenix, txt=text)

            labels = align_tokens_and_annotations_bilou(
                tokens, charix_to_tokenix_txt, annotations
            )
            # for token, label in zip(tokens, labels):
            #     print(token, "-", label)

            one_doc_json = {"id": str(ix), "tokens": tokens, "ner_tags": labels}
            manydoc_json.append(one_doc_json)
    return manydoc_json


def preprocess_and_split(annotations_and_texts_json):

    manydoc_json_filtered_bytype = [
        one_doc
        for one_doc in annotations_and_texts_json
        if "B-wikipedia" in one_doc["ner_tags"]
        or "U-wikipedia" in one_doc["ner_tags"]
    ]
    # filter length <= 512
    # manydoc_json_filtered_bytype_bylen = [
    #     one_doc for one_doc in manydoc_json_filtered_bytype if len(one_doc["tokens"]) <= 512
    # ]

    replacement = {
            "O":0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
            "B-MISC": 7,
            "I-MISC": 8,
    }
    
    for one_doc in manydoc_json_filtered_bytype:
        one_doc["ner_tags"] = [
            n if n in replacement.keys()[1:] or  else "O" for n in one_doc["ner_tags"]
        ]

    replacement = {
            "O":0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
            "B-MISC": 7,
            "I-MISC": 8,
    }

    for one_doc in manydoc_json_filtered_bytype:
        blub = [replacement[n] for n in one_doc["ner_tags"]]
        one_doc["ner_tags"] = blub

    train_size = 0.8
    test_size = 1 - train_size

    train_train_size = 0.8
    train_validate_size = 1 - train_train_size

    train, test = train_test_split(manydoc_json_filtered_bytype, train_size=train_size)

    train_train, train_validate = train_test_split(train, train_size=train_train_size)

    resulting_dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(pd.DataFrame(train_train), split="train"),
            "test": Dataset.from_pandas(pd.DataFrame(test), split="test"),
            "validation": Dataset.from_pandas(
                pd.DataFrame(train_validate), split="validation"
            ),
        }
    )

    # resulting_dataset = Dataset.from_dict(resulting_dataset_dict)  # TODO save as huggingface dataset

    return resulting_dataset_dict


def split_information(dataset_json):

    tmp_list_train = []
    for idx, t in enumerate(dataset_json["train"]):
        tmp = dataset_json["train"][idx]
        tmp["split"] = "train"
        tmp_list_train.append(tmp)

    tmp_list_test = []
    for idx, t in enumerate(dataset_json["test"]):
        tmp = dataset_json["test"][idx]
        tmp["split"] = "test"
        tmp_list_test.append(tmp)

    tmp_list_validation = []
    for idx, t in enumerate(dataset_json["validation"]):
        tmp = dataset_json["validation"][idx]
        tmp["split"] = "validation"
        tmp_list_validation.append(tmp)

    resulting_dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(pd.DataFrame(tmp_list_train), split="train"),
            "test": Dataset.from_pandas(pd.DataFrame(tmp_list_test), split="test"),
            "validation": Dataset.from_pandas(
                pd.DataFrame(tmp_list_validation), split="validation"
            ),
        }
    )

    return resulting_dataset_dict
