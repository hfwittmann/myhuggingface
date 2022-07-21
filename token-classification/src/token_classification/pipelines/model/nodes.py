"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.2
"""

import numpy as np
from datasets import Dataset
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


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


def setup(tokenized_datasets):
    model_name = L.model_checkpoint.rsplit("/", maxsplit=1)[-1]

    # TODO
    args = TrainingArguments(
        f"{model_name}-finetuned-{L.task}",
        evaluation_strategy="epoch",  # TODO
        learning_rate=2e-5,  # TODO
        per_device_train_batch_size=L.batch_size,  # TODO
        per_device_eval_batch_size=L.batch_size,  # TODO
        num_train_epochs=3,  # TODO
        weight_decay=0.01,  # TODO
        push_to_hub=False,  # TODO
    )

    data_collator = DataCollatorForTokenClassification(L.tokenizer)

    # use fraction for testing
    train_dataset = Dataset.from_dict(tokenized_datasets["train"][0:])  # TODO
    eval_dataset = Dataset.from_dict(tokenized_datasets["validation"][0:])  # TODO

    trainer = Trainer(
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
