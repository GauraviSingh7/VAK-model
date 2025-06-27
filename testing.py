import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# Load model and tokenizer
model_path = "./vak_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Prediction function
def predict_vak(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        vak_label = le.inverse_transform([pred_idx])[0]
        percentages = {label: round(prob.item() * 100, 2) for label, prob in zip(le.classes_, probs[0])}
        return vak_label, percentages

# 🔍 Sample test inputs
sentences = [
    "I understand better when I draw out my thoughts on paper.",
    "Talking things through with someone helps me process them.",
    "Let me try it myself a couple of times — that’s how I learn best.",
    "Just show me a diagram or chart, and I’ll figure it out.",
    "I prefer if someone explains it step by step out loud.",
    "I can’t follow this unless I actually do it with my own hands.",
    "Seeing a timeline really helps me remember history.",
    "Hearing a story helps me understand it more than reading.",
    "Practicing the presentation helps me more than just writing it.",
    "A visual summary or flowchart would make this much easier.",
    "I understand it best when I talk it through while sketching it",
    "well we are working on a tech-based project we hope to get the base models working by 3 months in accordance to our pipeline and by 6 months i hope we can finish our project"
]

# 🔁 Loop through and predict
for sentence in sentences:
    vak_label, percentages = predict_vak(sentence)
    print(f"\nInput: {sentence}")
    print(f"→ Predicted Style: {vak_label}")
    print("→ Probabilities:")
    for label, prob in percentages.items():
        print(f"   {label}: {prob}%")
