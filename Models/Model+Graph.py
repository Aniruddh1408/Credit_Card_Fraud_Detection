import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, log_loss, 
    precision_score, recall_score, roc_curve, auc, confusion_matrix, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# Handle physical core warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

# Load environment variables
load_dotenv()
user = "root"
password = os.getenv("SECURE_KEY")
database = "creditcardfraud"

# Load data
engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@localhost/{database}")
df = pd.read_sql("SELECT * FROM transactions", engine)

# Features and labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# LightGBM dataset
train_data = lgb.Dataset(X_train_scaled, label=y_train_smote)

# Parameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "n_jobs": 2,
    "random_state": 42
}

# Train model
model = lgb.train(params, train_data, num_boost_round=150)

# Predict and evaluate
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob >= 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {roc_auc:.2f}")
print("Log Loss:", log_loss(y_test, y_pred_prob))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict on single transaction
new_transaction = np.array([X_test_scaled[0]])
single_pred = model.predict(new_transaction)
print("\nNew Transaction Prediction:", "Fraud" if single_pred[0] >= 0.5 else "Not Fraud")

# # Start a figure with subplots
# fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle("Fraud Detection Model - Evaluation Metrics", fontsize=16)

# # 1. Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(ax=axs[0, 0], cmap="Blues", colorbar=False)
# axs[0, 0].set_title("Confusion Matrix")

# # 2. ROC Curve + AUC
# fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
# roc_auc = auc(fpr, tpr)
# axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
# axs[0, 1].plot([0, 1], [0, 1], 'k--')
# axs[0, 1].set_xlim([0.0, 1.0])
# axs[0, 1].set_ylim([0.0, 1.05])
# axs[0, 1].set_xlabel('False Positive Rate')
# axs[0, 1].set_ylabel('True Positive Rate')
# axs[0, 1].set_title('ROC Curve')
# axs[0, 1].legend(loc="lower right")

# # 3. Precision-Recall Curve
# precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
# axs[1, 0].plot(recall_vals, precision_vals, color='blue')
# axs[1, 0].set_title('Precision-Recall Curve')
# axs[1, 0].set_xlabel('Recall')
# axs[1, 0].set_ylabel('Precision')

# # 4. Feature Importance (LightGBM)
# importance = model.feature_importance()
# feature_names = df.drop(columns=["Class"]).columns
# sorted_idx = np.argsort(importance)[::-1][:10]
# axs[1, 1].barh(range(10), importance[sorted_idx][::-1])
# axs[1, 1].set_yticks(range(10))
# axs[1, 1].set_yticklabels(feature_names[sorted_idx][::-1])
# axs[1, 1].set_title("Top 10 Feature Importances")
# axs[1, 1].invert_yaxis()

# # Adjust layout
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# # Save LightGBM model
# model.save_model("C:/Users/DELL/fraud_detection_api/light_gbm.txt")


# # Save the scaler
# joblib.dump(scaler, "C:/Users/DELL/fraud_detection_api/scaler.pkl")

# # Save feature names (for correct input order during prediction)
# feature_names = X_train.columns.tolist()
# with open("C:/Users/DELL/fraud_detection_api/feature_names.pkl", "wb") as f:
#     joblib.dump(feature_names, f)
