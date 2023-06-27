from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback

from evaluate import load
import numpy as np

from properties import sequences_key, model_checkpoint, dataset, num_classes

metric = load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples[sequences_key], truncation=True)


tokenized_train = dataset["train"].map(preprocess_function, batched=True)
tokenized_val = dataset["validation"].map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)

training_args = TrainingArguments(
    output_dir="./results/",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    eval_steps=500,
    evaluation_strategy="steps",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()
