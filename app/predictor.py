import pandas as pd

def predict_diagnosis_and_mmse(model_clf, model_reg, patient_data, scaler):
    if isinstance(patient_data, dict):
        patient_data = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        patient_data = pd.DataFrame([patient_data])
    elif not isinstance(patient_data, pd.DataFrame):
        raise ValueError("Input patient_data must be a dict, Series, or DataFrame.")

    scaled = scaler.transform(patient_data)
    diagnosis = model_clf.predict(scaled)[0]
    mmse = model_reg.predict(scaled)[0]
    return diagnosis, mmse

def predict_mmse_hybrid(model, important_feats, full_patient_row):
    sample = full_patient_row[important_feats].values.reshape(1, -1)
    return model.predict(sample)[0]

def predict_mmse_xgb_rf(model, important_feats, full_patient_row):
    sample = full_patient_row[important_feats].values.reshape(1, -1)
    return model.predict(sample)[0]
