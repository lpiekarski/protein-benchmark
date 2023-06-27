from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from evaluate import load
import numpy as np

metric = load("accuracy")

model_checkpoint = "facebook/esm2_t12_35M_UR50D"
dataset = load_dataset("proteinea/solubility")
sequences_key = "sequences"
labels_key = "labels"
num_classes = 2


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples[sequences_key], truncation=True)


tokenized_train = dataset["train"].map(preprocess_function, batched=True)
tokenized_test = dataset["test"].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)

training_args = TrainingArguments(
    output_dir="./results/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
