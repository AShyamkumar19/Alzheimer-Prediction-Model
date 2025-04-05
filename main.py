from utils.preprocessing import load_and_clean_data, preprocess_data, split_data
from models.training import train_models, evaluate_models, show_feature_importance, shap_explanation
from app.predictor import predict_diagnosis_and_mmse

# Load and preprocess
df = load_and_clean_data("data/alzheimers_disease_data.csv")
X, y_class, y_reg, scaler = preprocess_data(df)
Xc_train, Xc_test, yc_train, yc_test, Xr_train, Xr_test, yr_train, yr_test = split_data(X, y_class, y_reg)

# Train models
rf_clf, xgb_clf, rf_reg, xgb_reg = train_models(Xc_train, yc_train, Xr_train, yr_train)

# Evaluate models
evaluate_models(rf_clf, xgb_clf, rf_reg, xgb_reg, Xc_test, yc_test, Xr_test, yr_test)

# Feature Importance & SHAP
print("\nFeature Importance for XGBoost:")
show_feature_importance(xgb_clf, Xc_train)
print("\nSHAP Summary Plot:")
shap_explanation(xgb_clf, Xc_test[:200])  

# Predict on a sample
sample = Xc_test.iloc[0]
prediction = predict_diagnosis_and_mmse(xgb_clf, xgb_reg, sample)
print(f"\nAgent Prediction for Sample Patient:\nDiagnosis = {'Alzheimer\'s' if prediction[0] else 'No Alzheimer\'s'}\nMMSE Score = {round(prediction[1], 2)}")
