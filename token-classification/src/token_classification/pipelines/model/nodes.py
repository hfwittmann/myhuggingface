"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.2
"""

import numpy as np
from datasets import Dataset, load_dataset, load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# label_list = datasets["train"].features[f"{task}_tags"].feature.names
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# TODO Move : to params
task = "ner"  # TODO Should be one of "ner", "pos" or "chunk" :TODO
model_checkpoint = "distilbert-base-uncased"  # TODO
batch_size = 16  # TODO

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = load_metric("seqeval")


def __compute_metrics(p):
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


def setup(tokenized_datasets):
    model_name = model_checkpoint.split("/")[-1]

    # TODO
    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy="epoch",  # TODO
        learning_rate=2e-5,  # TODO
        per_device_train_batch_size=batch_size,  # TODO
        per_device_eval_batch_size=batch_size,  # TODO
        num_train_epochs=3,  # TODO
        weight_decay=0.01,  # TODO
        push_to_hub=False,  # TODO
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # use fraction for testing
    train_dataset = Dataset.from_dict(tokenized_datasets["train"][:30])  # TODO
    eval_dataset = Dataset.from_dict(tokenized_datasets["validation"][:30])  # TODO

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=__compute_metrics,
    )

    trainer = train(trainer)
    evaluation = evaluate(trainer)
    # trainer.model.save_model()

    return trainer.model, evaluation


def train(trainer):
    trainer.train()
    return trainer


def evaluate(trainer):
    evaluation = trainer.evaluate()
    return evaluation
