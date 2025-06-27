import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load test data
df = pd.read_csv("test.csv")
sentences = df["Sentence"].tolist()
y_true = df["Type"].tolist()

# Load model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained("./vak_model")
tokenizer = BertTokenizer.from_pretrained("./vak_model")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Predict
model.eval()
y_pred = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=1).item()
        y_pred.append(le.inverse_transform([pred_label])[0])

# Classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("VAK Confusion Matrix (Evaluation on test.csv)")
plt.show()
