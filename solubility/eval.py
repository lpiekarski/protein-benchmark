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

model_path = "facebook/esm2_t6_8M_UR50D"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)

# Define test trainer
test_trainer = Trainer(
    model,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

model_predictions = test_trainer.predict(tokenized_test)

print(f"{model_predictions.metrics}")
with open("eval.txt", "w") as f:
    f.write(f"{model_predictions.metrics}")
