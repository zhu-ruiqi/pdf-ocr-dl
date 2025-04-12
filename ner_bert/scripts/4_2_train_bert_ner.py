import json
import os
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                          DataCollatorForTokenClassification)
from seqeval.metrics import classification_report
import numpy as np

# ==== Step 1: Load labels ====
label_list_path = "pdf-ocr-dl/ner_bert/config/label_list.txt"
with open(label_list_path, "r") as f:
    label_list = [line.strip() for line in f if line.strip()]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# ==== Step 2: Load dataset ====
data_files = {
    "train": "pdf-ocr-dl/ner_bert/data/train.json",
    "validation": "pdf-ocr-dl/ner_bert/data/dev.json"
}

raw_datasets = load_dataset("json", data_files=data_files)

# ==== Step 3: Tokenization ====
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label_to_id.get(example["ner_tags"][word_idx], 0))
        else:
            label_ids.append(label_to_id.get(example["ner_tags"][word_idx], 0))
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=False)

# ==== Step 4: Load model ====
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# ==== Step 5: Training arguments ====
args = TrainingArguments(
    output_dir="pdf-ocr-dl/ner_bert/model",
    evaluation_strategy="epoch",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    report_to="none"
)

# ==== Step 6: Trainer ====
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id_to_label[label] for label, pred in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]
    true_preds = [
        [id_to_label[pred] for label, pred in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]

    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# ==== Step 7: Train ====
trainer.train()
trainer.save_model("pdf-ocr-dl/ner_bert/model")
print("[INFO] Training complete! Model saved to: ner_bert/model")