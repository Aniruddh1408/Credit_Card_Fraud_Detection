Credit Card Fraud Detection System
Overview:

The Credit Card Fraud Detection System is an end-to-end machine learning application that detects fraudulent transactions in real-time. The system leverages machine learning models, a backend API, and a user-friendly frontend to simulate, predict, and visualize transaction risk.
It is ideal for understanding imbalanced datasets, fraud detection modeling, and full-stack ML deployment.

Features:

1. Real-time transaction simulation: Test transactions as normal or fraudulent.
2. High-accuracy fraud detection: Uses LightGBM with SMOTE and threshold tuning.
3. RESTful API: Built with FastAPI to handle prediction requests.
4. Interactive frontend: Built with Streamlit, showing transaction risk probabilities.
5. Database integration: Uses MySQL for storing transaction history.
6. Visualization: Graphical risk indicators and automated result display.

Project Structure:
CreditCardFraudDetection/
│
├── analysis/                 # Jupyter notebooks for data analysis
├── model_training/           # Scripts for training and saving ML models
│   ├── light_gbm.txt         # Trained LightGBM model
│   └── scaler.pkl            # StandardScaler for preprocessing
├── api_frontend/             # FastAPI backend and Streamlit frontend
│   ├── app.py                # Main FastAPI backend
│   ├── streamlit_app.py      # Streamlit frontend for user interaction
│   └── feature_names.pkl     # Feature list used by the model
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies

Technologies Used:

1. Python 3.x
2. LightGBM for fraud detection
3. Imbalanced-learn (SMOTE) for handling class imbalance
4. FastAPI for backend API
5. Streamlit for interactive frontend
6. MySQL for storing transaction data
7. Pandas, NumPy, Scikit-learn for data processing

How It Works:

1. Data preprocessing: Transactions are scaled using StandardScaler and features selected.
2. Model prediction: LightGBM predicts the probability of fraud.
3. Threshold tuning: Adjusted for optimal precision and recall.

   Future Enhancements

Deploy to cloud platforms for fully live predictions.

Implement multiple model ensembles to improve accuracy.

Add user authentication and historical transaction analytics.

Integrate with live transaction systems for real-world testing.

Author:
Aniruddh S.
GitHub: https://github.com/<your-username>

API handling: FastAPI receives transaction data and returns risk score.

Frontend display: Streamlit shows transaction status and risk probability in real-time.
