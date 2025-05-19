# Bank Customer Churn Prediction


**Bank Customer Churn Prediction**
This project builds and evaluates machine learning models to predict customer churn in a banking dataset using Logistic Regression, Random Forest, and XGBoost. It includes preprocessing steps, model training, and evaluation with full experiment tracking using MLflow.

**Project Structure**
churn-prediction/
│
├── dataset/
│   └── Churn_Modelling.csv
│
├── churn_model.py
├── README.md
├── requirements.txt
└── mlruns/ (created by MLflow after first run)

**How It Works**
Preprocessing
	•	Selects relevant features.
	•	Balances the dataset using downsampling.
	•	Splits the data into train and test sets.
	•	Applies:
	•	Standard Scaling to numerical columns.
	•	One-Hot Encoding to categorical columns.

Models Trained
	•	Logistic Regression
	•	Random Forest
	•	XGBoost

Each model is evaluated using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	Confusion Matrix

MLflow Tracking
	•	Logs parameters, metrics, models, and confusion matrix plots.
	•	Supports model comparison and reproducibility.


**How to Run**
1. Clone the Repository
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

2. Set Up the Environment
pip install -r requirements.txt

3. Start MLflow Server
mlflow ui
Visit http://127.0.0.1:5000 in your browser.

4. Run the Pipeline
python churn_model.py

This will:
	•	Preprocess data
	•	Train all 3 models
	•	Log results in MLflow


