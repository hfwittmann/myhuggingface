"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.2
"""

from enum import Enum
from lib2to3.pgen2 import token
from lib2to3.pgen2.token import tok_name
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset, load_metric
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .mysadice import SelfAdjDiceLoss
from functools import partial


def load_language_model(parameters):
    class LanguageModel:
        def __init__(self, parameters) -> None:
            # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # TODO
            # TODO Move : to params
            self.task = "ner"  # TODO Should be one of "ner", "pos" or "chunk" :TODO
            self.batch_size = parameters["batch_size"]  # TODO
            self.label_all_tokens = True  # TODO
            self.metric = load_metric(
                "/home/hfwittmann/Sync/mygrive/Colab Notebooks/myhuggingface/token-classification/src/token_classification/seqeval.py"
            )

            self.model_checkpoint = parameters[
                "model_checkpoint"
            ]  # "distilbert-base-german-cased"  # "t5-small"  # google/electra-small-discriminator # "deepset/gelectra-large"  # "deepset/gelectra-base"  # "distilbert-base-german-cased"  # TODO
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)  # TODO

            # label_list = datasets["train"].features[f"{task}_tags"].feature.names
            self.label_list = parameters["label_list"]

            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_checkpoint, num_labels=len(self.label_list)
            )  # .to(self.device)

    # L = LanguageModel(parameters)
    # class LanguageModel:
    #     def __init__(self) -> None:
    #         # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # TODO
    #         # TODO Move : to params
    #         self.task = "ner"  # TODO Should be one of "ner", "pos" or "chunk" :TODO
    #         self.batch_size = 16  # TODO
    #         self.label_all_tokens = True  # TODO
    #         self.metric = load_metric(
    #             "/home/hfwittmann/Sync/mygrive/Colab Notebooks/myhuggingface/token-classification/src/token_classification/seqeval.py"
    #         )

    #         self.model_checkpoint = "distilbert-base-german-cased"  # "t5-small"  # google/electra-small-discriminator # "deepset/gelectra-large"  # "deepset/gelectra-base"  # "distilbert-base-german-cased"  # TODO
    #         self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)  # TODO

    #         # label_list = datasets["train"].features[f"{task}_tags"].feature.names
    #         self.label_list = ["O", "B-DateOfBirth", "I-DateOfBirth", "U-DateOfBirth"]

    #         self.model = AutoModelForTokenClassification.from_pretrained(
    #             self.model_checkpoint, num_labels=len(self.label_list)
    #         )  # .to(self.device)

    L = LanguageModel(parameters)

    return L


def __compute_metrics(p, label_list, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def setup(L, tokenized_datasets, sample_train, sample_validation, model):
    model_name = L.model_checkpoint.rsplit("/", maxsplit=1)[-1]

    my_compute_metrics = partial(__compute_metrics, label_list=L.label_list, metric=L.metric)

    # TODO
    args = TrainingArguments(f"{model_name}-finetuned-{L.task}", **model)

    data_collator = DataCollatorForTokenClassification(L.tokenizer)

    # use fraction for testing
    import random

    if sample_train < len(tokenized_datasets["train"]) and sample_validation < len(tokenized_datasets["validation"]):
        """
        if there is more than enough data, only look at a smaller sample
        """

        train_df = pd.DataFrame(tokenized_datasets["train"])
        validation_df = pd.DataFrame(tokenized_datasets["validation"])

        train_samples = train_df.sample(n=sample_train, random_state=0)
        validation_samples = validation_df.sample(n=sample_validation, random_state=0)

        train_dataset = Dataset.from_pandas(train_samples)
        eval_dataset = Dataset.from_pandas(validation_samples)
    else:
        """
        if there is little data look at all of it
        """
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

    test_dataset = tokenized_datasets["test"]

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")

            criterion = SelfAdjDiceLoss(reduction="none", ignore_index=-100)

            loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            loss = loss.mean()
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        L.model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=L.tokenizer,
        compute_metrics=my_compute_metrics,
    )

    trainer.train()  # trainer = train(trainer)
    evaluation = trainer.evaluate()  # evaluation = evaluate(trainer)
    # trainer.model.save_model()
    predictions = trainer.predict(test_dataset)

    import pandas as pd

    Path("data/blab_folder").mkdir(exist_ok=True)

    for i in range(len(tokenized_datasets["test"])):
        tokens = [L.tokenizer.decode(t) for t in tokenized_datasets["test"][i]["input_ids"]]
        labels = test_dataset[i]["labels"]
        predicted_labels = predictions.predictions.argmax(axis=2)[i]
        min_len = min([len(tokens), len(labels), len(predicted_labels)])

        # labels = labels[:min_len]
        # predicted_labels = predicted_labels[:min_len]
        # tokens = tokens[:min_len]

        pd.DataFrame({"labels": labels, "predicted_labels": predicted_labels, "tokens": tokens}).to_csv(
            f"data/blab_folder/blab_{i}.csv"
        )

    return trainer.model, evaluation


def train(trainer):
    trainer.train()
    return trainer


def evaluate(trainer):
    evaluation = trainer.evaluate()
    return evaluation
