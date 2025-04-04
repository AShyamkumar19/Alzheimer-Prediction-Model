import pandas as pd

def load_and_clean_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.drop(columns=["DoctorInCharge"])
    data = data.dropna()
    
    return data


print(load_and_clean_data(r"data\alzheimers_disease_data.csv"))