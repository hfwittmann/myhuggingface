"""
This is a boilerplate pipeline 'tokenize_data'
generated using Kedro 0.18.2
"""

from transformers import AutoTokenizer
import pandas as pd
from functools import partial

# task = "ner"  # Should be one of "ner", "pos" or "chunk" # TODO : move to yaml
# model_checkpoint = "/shared/models/huggingface/transformers/distilbert-base-german-cased"  # TODO : move to yaml
# batch_size = 16  # TODO : move to yaml
# label_all_tokens = True  # TODO : move to yaml

# from ...helpers import L


def tokenize_and_align_labels(examples, window_length_before, window_length_after):
    tokenized_inputs = L.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{L.task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if L.label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    assert examples["split"][0] in ["train", "validation", "test"]

    if examples["split"][0] in ["train", "validate"]:
        half_window_length = 12
        # i = 0
        import numpy as np

        for i in range(len(labels)):

            hits = np.array(range(len(labels[i])))[np.array(labels[i]) > 0]
            labels_i_with_mask = [-100] * len(labels[i])
            for h in hits:
                start = h - window_length_before
                stop = h + window_length_after
                start = max(0, start)
                stop = min(len(labels[i]), stop)

                labels_i_with_mask[start:stop] = labels[i][start:stop]

            labels[i] = labels_i_with_mask

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def tokenize(dataset_json, window_length_before, window_length_after):

    # taal_dataset = dataset.map(tokenize_and_align_labels, batched=True) # works
    tokenize_and_align_labels_p = partial(
        tokenize_and_align_labels, window_length_before=window_length_before, window_length_after=window_length_after
    )
    taal_dataset_json = dataset_json.map(tokenize_and_align_labels_p, batched=True)  # works

    ## start : das hier gibt den fail !!!
    # taal_dataset_csv = dataset_csv.map(tokenize_and_align_labels, batched=True)
    ## end: das hier gibt den fail !!!

    return taal_dataset_json
