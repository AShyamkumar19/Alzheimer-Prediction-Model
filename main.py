from sklearn.metrics import mean_absolute_error
from utils.preprocessing import load_and_clean_data, preprocess_data, split_data
from models.training import train_models, evaluate_models, show_feature_importance, shap_explanation, hybrid_rf_xgb_pipeline, hybrid_xgb_rf_pipeline
from app.predictor import predict_diagnosis_and_mmse, predict_mmse_hybrid, predict_mmse_xgb_rf

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

sample_patient_num = 0
# Predict on a sample using XGBoost Model
sample = Xc_test.iloc[sample_patient_num]
prediction = predict_diagnosis_and_mmse(xgb_clf, xgb_reg, sample)
print(f"\nAgent Prediction for Sample Patient:\nDiagnosis = {'Alzheimer\'s' if prediction[0] else 'No Alzheimer\'s'}\nMMSE Score = {round(prediction[1], 2)}")

# RF Model
print("\nRF Model")
prediction_rf = predict_diagnosis_and_mmse(rf_clf, rf_reg, sample)
print(f"\nRF Model Prediction for Sample Patient:\nDiagnosis = {'Alzheimer\'s' if prediction_rf[0] else 'No Alzheimer\'s'}\nMMSE Score = {round(prediction_rf[1], 2)}")

# Optimum Output
print("\nOptimum Output")
prediction_optimum = predict_diagnosis_and_mmse(xgb_clf, rf_reg, sample)
print(f"\nOptimum Output Prediction for Sample Patient:\nDiagnosis = {'Alzheimer\'s' if prediction_optimum[0] else 'No Alzheimer\'s'}\nMMSE Score = {round(prediction_optimum[1], 2)}")

# Hybrid RF-XGB pipeline
hybrid_model, hybrid_feats = hybrid_rf_xgb_pipeline(Xr_train, yr_train, Xr_test, yr_test)

# Evaluate hybrid model
print("\nHybrid RF-XGB pipeline")
hybrid_model_rf_xgb, hybrid_feats_rf_xgb = hybrid_rf_xgb_pipeline(Xr_train, yr_train, Xr_test, yr_test)
hybrid_sample_rf_xgb = Xr_test.iloc[sample_patient_num]
hybrid_mmse_rf_xgb = predict_mmse_hybrid(hybrid_model_rf_xgb, hybrid_feats_rf_xgb, hybrid_sample_rf_xgb)
print(f"\nHybrid RF->XGB MMSE Prediction for Sample Patient: {round(hybrid_mmse_rf_xgb, 2)}")
Xr_test_reduced_rf_xgb = Xr_test[hybrid_feats_rf_xgb]
print("[Evaluation] RF->XGB Full Test Set MAE:", mean_absolute_error(yr_test, hybrid_model_rf_xgb.predict(Xr_test_reduced_rf_xgb)))

# Hybrid XGB -> RF
hybrid_model_xgb_rf, hybrid_feats_xgb_rf = hybrid_xgb_rf_pipeline(Xr_train, yr_train, Xr_test, yr_test)
hybrid_sample_xgb_rf = Xr_test.iloc[sample_patient_num]
hybrid_mmse_xgb_rf = predict_mmse_xgb_rf(hybrid_model_xgb_rf, hybrid_feats_xgb_rf, hybrid_sample_xgb_rf)
print(f"\nHybrid XGB->RF MMSE Prediction for Sample Patient: {round(hybrid_mmse_xgb_rf, 2)}")
Xr_test_reduced_xgb_rf = Xr_test[hybrid_feats_xgb_rf]
print("[Evaluation] XGB->RF Full Test Set MAE:", mean_absolute_error(yr_test, hybrid_model_xgb_rf.predict(Xr_test_reduced_xgb_rf)))

print("\nGround truth MMSE from dataset:", Xr_test.iloc[sample_patient_num], yr_test.iloc[sample_patient_num])
