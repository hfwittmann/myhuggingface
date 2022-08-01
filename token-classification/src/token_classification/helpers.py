import numpy as np
from datasets import Dataset, load_dataset, load_metric
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import yaml


class LanguageModel:
    def __init__(self) -> None:
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # TODO
        # TODO Move : to params
        self.task = "ner"  # TODO Should be one of "ner", "pos" or "chunk" :TODO
        self.batch_size = 16  # TODO
        self.label_all_tokens = True  # TODO
        self.metric = load_metric(
            "/home/hfwittmann/Sync/mygrive/Colab Notebooks/myhuggingface/token-classification/src/token_classification/seqeval.py"
        )

        self.model_checkpoint = "t5-small"  # google/electra-small-discriminator # "deepset/gelectra-large"  # "deepset/gelectra-base"  # "distilbert-base-german-cased"  # TODO
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)  # TODO

        # label_list = datasets["train"].features[f"{task}_tags"].feature.names
        self.label_list = ["O", "B-DateOfBirth", "I-DateOfBirth", "U-DateOfBirth"]

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.label_list)
        )  # .to(self.device)


L = LanguageModel()
