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


class LanguageModel:
    def __init__(self) -> None:
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"  # TODO
        # TODO Move : to params
        self.task = "ner"  # TODO Should be one of "ner", "pos" or "chunk" :TODO
        self.batch_size = 16  # TODO
        self.label_all_tokens = True  # TODO
        self.metric = load_metric("/shared/models/huggingface/transformers/metrics/seqeval/seqeval.py")

        self.model_checkpoint = "/shared/models/huggingface/transformers/distilbert-base-german-cased"  # TODO
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)  # TODO

        # label_list = datasets["train"].features[f"{task}_tags"].feature.names
        self.label_list = [
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC",
        ]

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.label_list)
        ).to(self.device)


L = LanguageModel()
