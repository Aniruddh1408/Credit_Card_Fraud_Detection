import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score   
from datetime import datetime

load_dotenv()
user = "root" 
password = os.getenv("SECURE_KEY") 
database = "creditcardfraud"

engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@localhost/{database}")
query = "SELECT * FROM transactions"
df = pd.read_sql(query, engine)


# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

# Plot histogram for Legitimate Transactions (Class = 0)
df[df["Class"] == 0]["Amount"].hist(bins=100, ax=axes[0], color="blue", edgecolor="black")
axes[0].set_title("Legitimate Transactions")
axes[0].set_ylabel("Count")

# Plot histogram for Fraudulent Transactions (Class = 1)
df[df["Class"] == 1]["Amount"].hist(bins=50, ax=axes[1], color="blue", edgecolor="black")
axes[1].set_title("Fraudulent Transactions")
axes[1].set_ylabel("Count")

# Set common labels
plt.xlabel("Transaction Amount")
plt.xticks(rotation=0)  # Keep x-axis labels readable
plt.tight_layout()

# Show the plot
plt.show()

