# End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API

📦 Telco Customer Churn Prediction – End-to-End ML Pipeline
This project builds a reusable and production-ready machine learning pipeline using the Scikit-learn Pipeline API to predict customer churn in a telecom company. The pipeline includes preprocessing, model training, hyperparameter tuning, and model export for future deployment.

🎯 Objective
Develop a complete ML pipeline to:

Preprocess raw customer data

Train and tune classification models

Package the pipeline for deployment using joblib

🧾 Dataset
Source: Telco Customer Churn Dataset
Or use the raw GitHub-hosted version:

python
Copy
Edit
url = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
Target Variable: Churn (Yes/No)

🛠️ Pipeline Components
🔹 1. Data Preprocessing
Implemented using sklearn.pipeline.Pipeline and ColumnTransformer:

Impute missing values

Encode categorical features using OneHotEncoder

Scale numerical features using StandardScaler

🔹 2. Model Training
Models trained:

LogisticRegression

RandomForestClassifier

🔹 3. Hyperparameter Tuning
Used GridSearchCV to find the best parameters.

Evaluation metrics: Accuracy and F1-Score

🔹 4. Model Export
Final pipeline exported using joblib for production reuse:

python
Copy
Edit
import joblib
joblib.dump(best_pipeline, "churn_prediction_pipeline.joblib")
📈 Evaluation Metrics
The pipeline reports:

Accuracy

F1-score

Confusion matrix

Example:

python
Copy
Edit
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
🚀 How to Run
Clone this repository

Install dependencies:

bash
Copy
Edit
pip install pandas scikit-learn joblib
Run the pipeline:

bash
Copy
Edit
python churn_pipeline.py
Load and reuse the trained model:

python
Copy
Edit
pipeline = joblib.load("churn_prediction_pipeline.joblib")
pipeline.predict(new_data)
📚 Skills Gained
🏗️ Scikit-learn Pipeline design

🔧 Hyperparameter tuning using GridSearchCV

📦 Exporting pipelines with joblib

⚙️ Best practices for production-ready ML

📂 Project Structure
bash
Copy
Edit
├── churn_pipeline.py         # Main pipeline script
├── README.md
├── requirements.txt
└── churn_prediction_pipeline.joblib   # Exported model
