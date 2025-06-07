import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore")

# # --- SETUP MLflow ---
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow UI
# mlflow.set_experiment("Attrition_Models")

# --- LOAD PREPROCESSED DATA ---
data_path = 'processed_data.csv'
df = pd.read_csv(data_path)

# --- SPLIT FEATURES AND TARGET ---
X = df.drop(columns=['Attrition'])
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- APPLY SMOTE ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --- TRACKING & TRAINING ---
mlflow.sklearn.autolog()  # Gunakan autolog untuk penilaian BASIC

with mlflow.start_run(run_name="SVM_Employee_Attrition_No_Hyperparameter_Tuning"):
    model = SVC()
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)

    print("--- Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

    classification_report = classification_report(y_test, y_pred)
    print(classification_report)
