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
print(df.head())
print(df['Class'].value_counts())

# Count occurrences of each class
class_counts = df['Class'].value_counts()


# Create a bar plot
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette=['blue', 'red'], legend=False) 


# Labels and title

plt.title("Credit Card Transactions: Fraud vs Non-Fraud") 
plt.show()   

pd.set_option("display.float_format", lambda x: "%.3f" % x)
print(df.describe())    

