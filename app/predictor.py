def predict_diagnosis_and_mmse(model_clf, model_reg, patient_data):
    diagnosis = model_clf.predict([patient_data])[0]
    mmse = model_reg.predict([patient_data])[0]
    return diagnosis, mmse
