import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

# --- Load Dataset ---
data_path = "processed_data.csv"
df = pd.read_csv(data_path)

# --- Split Features and Target ---
X = df.drop(columns=['Attrition'])
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Apply SMOTE ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --- Hyperparameters ---
C_param = 1.0
kernel_param = "rbf"

# --- MLflow Logging ---
input_example = X_train.head(5)

# --- Custom MLflow Tracking URI ---
remote_server_uri = "http://82.197.71.171:5000"
mlflow.set_tracking_uri(remote_server_uri)

with mlflow.start_run(run_name="SVM_Attrition_Manual_Log"):
    model = SVC(C=C_param, kernel=kernel_param, probability=True)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    # Log Parameters & Metrics
    mlflow.log_param("C", C_param)
    mlflow.log_param("kernel", kernel_param)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # --- Evaluation Output ---
    print("--- Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
