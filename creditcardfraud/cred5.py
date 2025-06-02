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


#Statistics with respect to time (hours)
if "Hour" not in df.columns:
    df["Hour"] = (df["Time"] // 3600) % 24  

# Function to compare statistics between Legitimate and Fraudulent transactions
def compare_leg_fraud(attribute):
    if attribute not in df.columns:
        raise KeyError(f"Column '{attribute}' not found in dataframe")
    
    # Extract data for legitimate and fraudulent transactions
    leg_trS = df.loc[df['Class'] == 0, attribute]  # Legitimate transactions
    frd_trS = df.loc[df['Class'] == 1, attribute]  # Fraudulent transactions
    
    # Generate summary statistics
    summary_df = pd.DataFrame({
        "Legitimate": leg_trS.describe(),
        "Fraudulent": frd_trS.describe()
    })

    return summary_df
summary = compare_leg_fraud('Hour')
print(summary)



    