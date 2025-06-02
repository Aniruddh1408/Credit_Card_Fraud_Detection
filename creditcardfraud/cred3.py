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
 

#Check all cases
ax=df.drop("Class",axis=1).hist(figsize=(10,12),bins=100)
for axis in ax.flatten():
    axis.set_xticklabels([])
    axis.set_yticklabels([])
plt.show()

