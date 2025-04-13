import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ✅ Step 1: Load CSVs
df_garantie = pd.read_csv("../swsgarantie.csv", sep=",", encoding="latin1", quotechar='"')
df_nietperse = pd.read_csv("../nietpersegarantie.csv", sep=",", encoding="latin1", quotechar='"')

# Combine datasets
original_df = pd.concat([df_garantie, df_nietperse], ignore_index=True)

# ✅ Step 2: Define class labels
label_mapping = {
    "Nee": 0,
    "Ja": 1,
    "1": 2,
    "3": 3,
    "6": 4,
    "12": 5
}
reverse_mapping = {v: k for k, v in label_mapping.items()}

# ✅ Step 3: Map labels
original_df["label"] = original_df["guarantee"].map(label_mapping).fillna(0).astype(int)

# ✅ Step 4: Load tokenizer
tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

# ✅ Step 5: Clean descriptions
def clean_description(text, max_tokens=512):
    tokens = tokenizer.encode(str(text), truncation=False)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    return tokenizer.decode(tokens)

original_df["description"] = original_df["description"].astype(str).apply(clean_description)

# ✅ Step 6: Train/val/test split
train_df, temp_df = train_test_split(original_df, test_size=0.2, random_state=42, stratify=original_df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# ✅ Step 7: Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[["description", "label"]])
val_dataset = Dataset.from_pandas(val_df[["description", "label"]])
test_dataset = Dataset.from_pandas(test_df[["description", "label"]])

# ✅ Step 8: Tokenize
def tokenize(batch):
    return tokenizer(batch["description"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ✅ Step 9: Load model
model = BertForSequenceClassification.from_pretrained("GroNLP/bert-base-dutch-cased", num_labels=len(label_mapping))

# ✅ Step 10: Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

# ✅ Step 11: TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_steps=500,
    save_total_limit=2
)

# ✅ Step 12: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ✅ Step 13: Train
trainer.train()

# ✅ Step 14: Evaluate on test set
print("\n📊 Evaluation on test set:")
test_results = trainer.evaluate(test_dataset)
print(test_results)

# ✅ Step 15: Predict on full dataset and export
full_dataset = Dataset.from_pandas(original_df[["description"]])
full_dataset = full_dataset.map(tokenize, batched=True)
full_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

predictions = trainer.predict(full_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)
original_df["predicted_label"] = predicted_labels
original_df["predicted_label_str"] = original_df["predicted_label"].map(reverse_mapping)

# ✅ Step 16: Save to CSV
original_df.to_csv("predicted_guarantees.csv", index=False)
print("\n✅ Predictions saved to predicted_guarantees.csv")