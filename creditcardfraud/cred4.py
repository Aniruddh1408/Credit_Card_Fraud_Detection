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


#Histogram to check Legitimate vs Fraudulent transactions wrt amount and time
df["Hour"] = (df["Time"] // 3600) % 24  # Convert seconds to hours (0-23)
df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

# Plot Legitimate Transactions
axes[0].hist(df[df["Class"] == 0]["Hour"], bins=24, range=(0, 24), color="blue", alpha=0.7, edgecolor="black")
axes[0].set_title("Legitimate Transactions")

# Plot Fraudulent Transactions
axes[1].hist(df[df["Class"] == 1]["Hour"], bins=24, range=(0, 24), color="red", alpha=0.7, edgecolor="black")
axes[1].set_title("Fraudulent Transactions")

plt.xlabel("Hour of the Day")
plt.xticks(range(24))
plt.tight_layout()
plt.show()

