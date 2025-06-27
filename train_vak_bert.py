import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import pickle

# Load dataset
df = pd.read_excel("VAK_dataset.xlsx")
texts = df["Sentence"].tolist()
labels = df["Type"].tolist()

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_labels = len(le.classes_)

# Save label encoder for later inference
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class VAKDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = VAKDataset(train_texts, train_labels, tokenizer)
val_dataset = VAKDataset(val_texts, val_labels, tokenizer)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vak_model",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and save model
trainer.train()
trainer.save_model("./vak_model")
tokenizer.save_pretrained("./vak_model")
