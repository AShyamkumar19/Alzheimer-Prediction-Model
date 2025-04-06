from sklearn.metrics import mean_absolute_error
from utils.preprocessing import load_and_clean_data, preprocess_data, split_data
from models.training import train_models, evaluate_models, show_feature_importance, shap_explanation, hybrid_rf_xgb_pipeline, hybrid_xgb_rf_pipeline, save_model_artifacts
from app.predictor import predict_diagnosis_and_mmse, predict_mmse_hybrid, predict_mmse_xgb_rf
from RL_Agent.mmse_rl_agent import MMSEEnv, QLearningAgent
from joblib import load
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

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

sample_patient_num = 9
# Predict on a sample using XGBoost Model
sample = Xc_test.iloc[sample_patient_num]
prediction = predict_diagnosis_and_mmse(xgb_clf, xgb_reg, sample)
print(f"\nAgent Prediction for Sample Patient:\nDiagnosis = {'Alzheimer\'s' if prediction[0] else 'No Alzheimer\'s'}\nMMSE Score = {round(prediction[1], 2)}")

'''
    Running RF and XGBoost Models and its hybrid 
'''

# # RF Model
# print("\nRF Model")
# prediction_rf = predict_diagnosis_and_mmse(rf_clf, rf_reg, sample)
# print(f"\nRF Model Prediction for Sample Patient:\nDiagnosis = {'Alzheimer\'s' if prediction_rf[0] else 'No Alzheimer\'s'}\nMMSE Score = {round(prediction_rf[1], 2)}")

# Optimum Output
print("\nOptimum Output")
prediction_optimum = predict_diagnosis_and_mmse(xgb_clf, rf_reg, sample)
print(f"\nOptimum Output Prediction for Sample Patient:\nDiagnosis = {'Alzheimer\'s' if prediction_optimum[0] else 'No Alzheimer\'s'}\nMMSE Score = {round(prediction_optimum[1], 2)}")

# # Hybrid RF-XGB pipeline
# hybrid_model, hybrid_feats = hybrid_rf_xgb_pipeline(Xr_train, yr_train, Xr_test, yr_test)

# # Evaluate hybrid model
# print("\nHybrid RF-XGB pipeline")
# hybrid_model_rf_xgb, hybrid_feats_rf_xgb = hybrid_rf_xgb_pipeline(Xr_train, yr_train, Xr_test, yr_test)
# hybrid_sample_rf_xgb = Xr_test.iloc[sample_patient_num]
# hybrid_mmse_rf_xgb = predict_mmse_hybrid(hybrid_model_rf_xgb, hybrid_feats_rf_xgb, hybrid_sample_rf_xgb)
# print(f"\nHybrid RF->XGB MMSE Prediction for Sample Patient: {round(hybrid_mmse_rf_xgb, 2)}")
# Xr_test_reduced_rf_xgb = Xr_test[hybrid_feats_rf_xgb]
# print("[Evaluation] RF->XGB Full Test Set MAE:", mean_absolute_error(yr_test, hybrid_model_rf_xgb.predict(Xr_test_reduced_rf_xgb)))

# # Hybrid XGB -> RF
# hybrid_model_xgb_rf, hybrid_feats_xgb_rf = hybrid_xgb_rf_pipeline(Xr_train, yr_train, Xr_test, yr_test)
# hybrid_sample_xgb_rf = Xr_test.iloc[sample_patient_num]
# hybrid_mmse_xgb_rf = predict_mmse_xgb_rf(hybrid_model_xgb_rf, hybrid_feats_xgb_rf, hybrid_sample_xgb_rf)
# print(f"\nHybrid XGB->RF MMSE Prediction for Sample Patient: {round(hybrid_mmse_xgb_rf, 2)}")
# Xr_test_reduced_xgb_rf = Xr_test[hybrid_feats_xgb_rf]
# print("[Evaluation] XGB->RF Full Test Set MAE:", mean_absolute_error(yr_test, hybrid_model_xgb_rf.predict(Xr_test_reduced_xgb_rf)))

print("\nGround truth MMSE from dataset:")
print("Patient Features:\n", Xr_test.iloc[sample_patient_num])
print("Actual MMSE Score:", yr_test.iloc[sample_patient_num])

'''
    Now Running the RL Agent
'''

save_model_artifacts(rf_reg, xgb_reg, scaler)

# Load the saved models (just in case but not needed)
rf_reg = load("models/rf_reg.joblib")
xgb_reg = load("models/xgb_reg.joblib")
scaler = load("models/scaler.joblib")

all_features = list(X.columns)
sample_dict = dict(zip(all_features, Xc_test.iloc[sample_patient_num]))

# Initial MMSE prediction (before agent acts)
initial_df = pd.DataFrame([sample_dict])[all_features]
initial_scaled = scaler.transform(initial_df)
initial_mmse = xgb_reg.predict(initial_scaled)[0] # can be changed to rf_reg.predict(initial_scaled)[0]

print("\nInitial MMSE Prediction (before action):", round(initial_mmse, 2))

env = MMSEEnv(model=xgb_reg, scaler=scaler, feature_names=all_features, initial_state=sample_dict) # can be changed to rf_reg.predict(initial_scaled)[0]
agent = QLearningAgent(actions=list(range(7)))

print("Training the agent...")
# for episode in range(1000):
#     state = env.reset(sample_dict)
#     action = agent.choose_action(state)
#     next_state, reward, done = env.step(action)
#     agent.learn(state, action, reward, next_state)
#print("Training complete!")

for episode in range(1000):
    state = env.reset(sample_dict)
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
    agent.decay_epsilon()
print("Training complete!")

# Get best action and apply it once more to observe its effect
best_action = agent.choose_action(sample_dict)
action_map = {
    0: "Do Nothing", 1: "Improve Sleep", 2: "Increase Activity", 3: "Improve Diet",
    4: "Reduce Alcohol", 5: "Manage BMI", 6: "Quit Smoking"
}

# Q-values
state_key = agent.get_state_key(sample_dict)
q_values = agent.q_table[state_key]

# Get top-k actions
k = 3  # you can change this to show more or fewer
top_actions = np.argsort(q_values)[::-1][:k]

print(f"\nTop {k} Recommended Actions and Expected MMSE Impact:")

for act in top_actions:
    # Reset environment before simulating
    env.reset(sample_dict)
    
    # Apply selected action
    _, reward, _ = env.step(act)
    
    # Get new MMSE after this action
    post_action_df = pd.DataFrame([env.state])[all_features]
    post_scaled = scaler.transform(post_action_df)
    post_mmse = xgb_reg.predict(post_scaled)[0]  # can be changed to rf_reg.predict(post_scaled)[0]
    
    print(f" -> {action_map[act]}:")
    print(f"     Predicted MMSE: {round(post_mmse, 2)}")
    print(f"     Change from baseline: {round(post_mmse - initial_mmse, 2)}")
