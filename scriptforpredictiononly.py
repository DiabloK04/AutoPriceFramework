import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# ✅ Step 1: Load raw CSV (❗️Removed label logic)
df = pd.read_csv("../bigdata2.csv", sep=",", encoding="latin1", quotechar='"')

# ✅ Step 2: Sample 25% of the data (✔️ NEW line)
df = df.sample(frac=0.25, random_state=42).reset_index(drop=True)

# ✅ Step 3: Load tokenizer and truncate descriptions
tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

def clean_description(text, max_tokens=512):
    tokens = tokenizer.encode(str(text), truncation=False)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    return tokenizer.decode(tokens)

df["description"] = df["description"].astype(str).apply(clean_description)

# ✅ Step 4: Convert to HuggingFace dataset (✔️ Removed label column)
dataset = Dataset.from_pandas(df[["description"]])

# ✅ Step 5: Tokenize
def tokenize(batch):
    return tokenizer(batch["description"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ✅ Step 6: Load trained model for prediction
model = BertForSequenceClassification.from_pretrained('./results/checkpoint-308', num_labels=6)

# ✅ Step 7: Setup Trainer (❗️ Only for prediction now)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args
)

# ✅ Step 8: Make predictions
print("\nMaking predictions on the 25% sample...")
predictions = trainer.predict(dataset)
predicted_labels = predictions.predictions.argmax(axis=-1)

# ✅ Step 9: Store predictions
df['predicted_label'] = predicted_labels
df.to_csv("predictions_only.csv", index=False)

print("✅ Predictions exported to 'predictions_only.csv'")
