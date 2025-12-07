import mlflow

mlflow.set_tracking_uri("http://3.110.192.209:5000")

print("Registering REGRESSION Model...")
mlflow.register_model(
    "s3://mlflow-tracking-loan/4/models/m-aa8bb88af9d74eaf9adc82fbb76f3081/artifacts",
    "EMI_Regression_XGBoost"
)

print("Registering CLASSIFICATION Model...")
mlflow.register_model(
    "s3://mlflow-tracking-loan/5/models/m-7dd42256b0ba4c8b9aa2dfd2f8fc6c41/artifacts",
    "EMI_Classification_XGBoost"
)

print("✅ BOTH MODELS REGISTERED SUCCESSFULLY")
