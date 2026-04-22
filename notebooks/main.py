import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split

print("Starting fraud detection program...\n")

# Load dataset
data = pd.read_csv("data.csv")

# Features and label
X = data.drop("Class", axis=1)
y = data["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Isolation Forest
# ---------------------------
iso_model = IsolationForest(contamination=0.002, random_state=42)
iso_model.fit(X_train)

iso_pred_raw = iso_model.predict(X_test)
iso_pred = [1 if x == -1 else 0 for x in iso_pred_raw]

print("Isolation Forest Evaluation:\n")
print(classification_report(y_test, iso_pred))

# Confusion Matrix - Isolation Forest
cm_iso = confusion_matrix(y_test, iso_pred)
disp_iso = ConfusionMatrixDisplay(confusion_matrix=cm_iso, display_labels=["Normal", "Fraud"])
disp_iso.plot()
plt.title("Confusion Matrix - Isolation Forest")
plt.savefig("confusion_matrix_isolation_forest.png", bbox_inches="tight")
plt.close()

# ---------------------------
# Random Forest
# ---------------------------
rf_model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Evaluation:\n")
print(classification_report(y_test, rf_pred))

# Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Normal", "Fraud"])
disp_rf.plot()
plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix_random_forest.png", bbox_inches="tight")
plt.close()

# ROC Curve - Random Forest
fpr, tpr, thresholds = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.savefig("roc_curve_random_forest.png", bbox_inches="tight")
plt.close()

print("\nVisuals saved successfully:")
print("- confusion_matrix_isolation_forest.png")
print("- confusion_matrix_random_forest.png")
print("- roc_curve_random_forest.png")

input("\nPress Enter to exit...")