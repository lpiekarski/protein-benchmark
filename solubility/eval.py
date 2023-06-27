from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback

from evaluate import load
import numpy as np

from properties import model_checkpoint, sequences_key, dataset, num_classes

metric = load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples[sequences_key], truncation=True)


tokenized_test = dataset["test"].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model_path = "output/checkpoint-50000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)

# Define test trainer
test_trainer = Trainer(model, compute_metrics=compute_metrics)

test_trainer.evaluate(tokenized_test)
