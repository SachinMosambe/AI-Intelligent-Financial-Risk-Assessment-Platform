from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://3.110.192.209:5000")

# ✅ Promote REGRESSION
client.transition_model_version_stage(
    name="EMI_Regression_XGBoost",
    version=1,
    stage="Production",
    archive_existing_versions=True
)

# ✅ Promote CLASSIFICATION
client.transition_model_version_stage(
    name="EMI_Classification_XGBoost",
    version=1,
    stage="Production",
    archive_existing_versions=True
)

print("✅ BOTH MODELS ARE NOW IN PRODUCTION")
