from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback
from datasets import load_dataset
from evaluate import load
import numpy as np


def eval_seq_classification_task(model_checkpoint, gradient_accumulation, dataset_name, sequences_key, num_classes, has_validation=False):
    dataset = load_dataset(dataset_name)

    metric = load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        return tokenizer(examples[sequences_key], truncation=True)

    if has_validation:
        tokenized_train = dataset["train"].map(preprocess_function, batched=True)
        tokenized_val = dataset["validation"].map(preprocess_function, batched=True)
    else:
        train_data, test_data = train_test_split(dataset["train"], test_size=0.15)
        tokenized_train = train_data.map(preprocess_function, batched=True)
        tokenized_val = test_data.map(preprocess_function, batched=True)

    tokenized_test = dataset["test"].map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)

    training_args = TrainingArguments(
        output_dir=f"./trained_models/{model_checkpoint}",
        learning_rate=2e-5,
        per_device_train_batch_size=64 // gradient_accumulation,
        per_device_eval_batch_size=64 // gradient_accumulation,
        num_train_epochs=2,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        eval_steps=2000 // gradient_accumulation,
        save_steps=2000 // gradient_accumulation,
        logging_steps=500 // gradient_accumulation,
        evaluation_strategy="steps",
        fp16=True,
        gradient_accumulation_steps=gradient_accumulation
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
    )

    trainer.train()

    test_trainer = Trainer(
        model,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    model_predictions = test_trainer.predict(tokenized_test)

    print(f"Evaluation metrics: {model_predictions.metrics}")