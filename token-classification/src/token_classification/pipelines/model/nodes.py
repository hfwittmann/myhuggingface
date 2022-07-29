"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.2
"""

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from .mysadice import SelfAdjDiceLoss
from torch import nn
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from ...helpers import L


def __compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [L.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [L.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = L.metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def setup(tokenized_datasets, sample_train, sample_validation):
    model_name = L.model_checkpoint.rsplit("/", maxsplit=1)[-1]

    # TODO
    args = TrainingArguments(
        f"{model_name}-finetuned-{L.task}",
        evaluation_strategy="epoch",  # TODO
        learning_rate=1e-5,  # TODO
        per_device_train_batch_size=L.batch_size,  # TODO
        per_device_eval_batch_size=L.batch_size,  # TODO
        num_train_epochs=5,  # TODO
        weight_decay=0.01,  # TODO
        push_to_hub=False,  # TODO
    )

    data_collator = DataCollatorForTokenClassification(L.tokenizer)

    # use fraction for testing
    import random

    train_df = pd.DataFrame(tokenized_datasets["train"])
    validation_df = pd.DataFrame(tokenized_datasets["validation"])

    train_samples = train_df.sample(n=sample_train, random_state=0)
    validation_samples = validation_df.sample(n=sample_validation, random_state=0)

    train_dataset = Dataset.from_pandas(train_samples)
    eval_dataset = Dataset.from_pandas(validation_samples)

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
        compute_metrics=__compute_metrics,
    )

    trainer.train()  # trainer = train(trainer)
    evaluation = trainer.evaluate()  # evaluation = evaluate(trainer)
    # trainer.model.save_model()

    return trainer.model, evaluation


def train(trainer):
    trainer.train()
    return trainer


def evaluate(trainer):
    evaluation = trainer.evaluate()
    return evaluation
