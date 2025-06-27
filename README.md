# VAK Learning Style Classifier

This project implements a **VAK (Visual, Auditory, Kinesthetic)** learning style classifier using a fine-tuned BERT model. It predicts a user's preferred learning style based on their conversational input, making it ideal for use in personalized coaching or adaptive learning systems.

---

## 🚀 Features
- Fine-tuned BERT model (HuggingFace Transformers)
- Trained on 3000+ realistic coachee-style utterances
- Classifies input text as Visual, Auditory, or Kinesthetic
- Supports batch evaluation and visualization (confusion matrix, F1-score)
- Ready for integration into real-time coaching systems

---

## 📁 Project Structure
```
.
├── vak_model/                # Folder to store the fine-tuned BERT model (not included due to size)
├── VAK_dataset.xlsx          # Training dataset (avl on request)
├── test.csv                  # Realistic test dataset (sentences + true labels)
├── label_encoder.pkl         # LabelEncoder for decoding predicted class
├── train_vak_bert.py         # Script to train the model
├── eval_report.py            # Generates classification report and confusion matrix
├── testing.py                # Batch test script for predictions
```

---

## 🧠 Dataset Overview
The training data contains:
- Clear, ambiguous, and subtle V/A/K phrases
- Balanced classes (Visual, Auditory, Kinesthetic)
- Augmented examples and paraphrased utterances

Format:
```
Sentence,Type
"I remember best when I see a diagram.",Visual
"Talking through steps helps me remember.",Auditory
"I need to try it out to understand.",Kinesthetic
```

---

## 🛠️ How to Train
1. Install dependencies:
```bash
pip install torch transformers scikit-learn pandas matplotlib
```
2. Run training:
```bash
python train_vak_bert.py
```
3. Model will be saved in the `vak_model/` folder (create if not present).

---

## 📊 How to Evaluate
Use the test set (`test.csv`) and run:
```bash
python eval_report.py
```
This outputs:
- Precision, recall, F1-score per class
- Confusion matrix

---

## 🔍 Sample Output
```
Accuracy: 0.88
Kinesthetic Recall: 0.86
Visual F1: 0.90
Auditory F1: 0.88
```

---

## 📦 Future Work
- Support soft labels / multi-label classification
- Real-time transcription + classification pipeline

