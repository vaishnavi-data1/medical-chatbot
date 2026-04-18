import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.decomposition import TruncatedSVD

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
DATASET_PATH = "dataset/Training.csv"
MODEL_DIR = "model_files"
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pkl")

df = pd.read_csv(DATASET_PATH)

required_cols = ["Symptoms", "Disease"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df = df.dropna(subset=required_cols)
df["Symptoms"] = df["Symptoms"].str.lower()
df["Disease"] = df["Disease"].str.lower().str.strip()

# Remove rare diseases (noise & overfitting control)
valid_diseases = df["Disease"].value_counts()
valid_diseases = valid_diseases[valid_diseases >= 5].index
df = df[df["Disease"].isin(valid_diseases)]

X = df["Symptoms"]
y = df["Disease"]

# -------------------------------------------------
# TF-IDF VECTORIZATION
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

# -------------------------------------------------
# TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# MODELS (ANTI OVER/UNDERFITTING)
# -------------------------------------------------

# Logistic Regression – strong regularization
lr = LogisticRegression(
    max_iter=4000,
    C=0.8,                 # regularization
    class_weight="balanced",
    solver="lbfgs"
)

# SVM – margin based (good generalization)
svm = SVC(
    kernel="linear",
    C=1.0,
    probability=True,
    class_weight="balanced"
)

# Random Forest – controlled depth
rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,          # prevents overfitting
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)

# ENSEMBLE
model = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("svm", svm),
        ("rf", rf)
    ],
    voting="soft"
)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
print("🚀 Training ensemble model...")
model.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="weighted"
)

print("\n📊 MODEL PERFORMANCE")
print("-------------------------")
print(f"Accuracy : {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall   : {rec * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------------------------
# CORRELATION MATRIX (TF-IDF REDUCED)
# -------------------------------------------------
svd = TruncatedSVD(n_components=20, random_state=42)
X_reduced = svd.fit_transform(X_vec.toarray())

corr_matrix = np.corrcoef(X_reduced.T)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Feature Correlation Matrix (Reduced TF-IDF)")
plt.show()

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        f
    )

print("\n✅ Model saved successfully at:", MODEL_PATH)

