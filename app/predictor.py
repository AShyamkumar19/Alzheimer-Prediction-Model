def predict_diagnosis_and_mmse(model_clf, model_reg, patient_data):
    diagnosis = model_clf.predict([patient_data])[0]
    mmse = model_reg.predict([patient_data])[0]
    return diagnosis, mmse

def predict_mmse_hybrid(model, important_feats, full_patient_row):
    sample = full_patient_row[important_feats].values.reshape(1, -1)
    return model.predict(sample)[0]

def predict_mmse_xgb_rf(model, important_feats, full_patient_row):
    sample = full_patient_row[important_feats].values.reshape(1, -1)
    return model.predict(sample)[0]
