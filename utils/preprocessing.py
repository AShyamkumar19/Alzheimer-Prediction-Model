import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.drop(columns=["PatientID", "DoctorInCharge"])
    data = data.dropna()    
    
    return data

def preprocess_data(data):
    scaler = StandardScaler()
    X = data.drop(columns=["Diagnosis", "MMSE"])
    y_diagnosis = data["Diagnosis"]
    y_mmse = data["MMSE"]
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y_diagnosis, y_mmse, scaler

def split_data(X, y_diagnosis, y_mmse):
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(X, y_diagnosis, test_size=0.5, random_state=42)
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(X, y_mmse, test_size=0.5, random_state=42)
    return Xd_train, Xd_test, yd_train, yd_test, Xm_train, Xm_test, ym_train, ym_test
