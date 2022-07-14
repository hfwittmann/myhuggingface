"""
This is a boilerplate pipeline 'tokenize_data'
generated using Kedro 0.18.2
"""

from transformers import AutoTokenizer
import pandas as pd

task = "ner"  # Should be one of "ner", "pos" or "chunk" # TODO : move to yaml
model_checkpoint = "distilbert-base-uncased"  # TODO : move to yaml
batch_size = 16  # TODO : move to yaml
label_all_tokens = True  # TODO : move to yaml

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
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
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tokenize(dataset, dataset_csv, dataset_json):

    # taal_dataset = dataset.map(tokenize_and_align_labels, batched=True) # works
    taal_dataset_json = dataset_json.map(tokenize_and_align_labels, batched=True)  # works

    ## start : das hier gibt den fail !!!
    # taal_dataset_csv = dataset_csv.map(tokenize_and_align_labels, batched=True)
    ## end: das hier gibt den fail !!!

    return taal_dataset_json
