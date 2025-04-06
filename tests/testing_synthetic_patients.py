from models.training import load_model_and_scaler
from RL_Agent.mmse_rl_agent import MMSEEnv, QLearningAgent
from app.predictor import predict_diagnosis_and_mmse
from utils.preprocessing import load_and_clean_data, preprocess_data
import pandas as pd
import numpy as np


df = load_and_clean_data("data/alzheimers_disease_data.csv")
X, y_class, y_reg, scaler = preprocess_data(df)

# Load trained model and scaler (assumes you've saved them)
rf_reg, scaler = load_model_and_scaler()

# Define features from training
print("\n--- Testing Synthetic High-Risk Patient ---")

# # Use full feature list for consistency
# all_features = list(X.columns)

# # Step 1: Define high-risk patient
# synthetic_patient = {
#     "Age": 80,
#     "Gender": 1,
#     "Ethnicity": 1,
#     "EducationLevel": 0,
#     "BMI": 38,
#     "Smoking": 1,
#     "AlcoholConsumption": 18,
#     "PhysicalActivity": 0,
#     "DietQuality": 1,
#     "SleepQuality": 4,
#     "FamilyHistoryAlzheimers": 1,
#     "CardiovascularDisease": 1,
#     "Diabetes": 1,
#     "Depression": 1,
#     "HeadInjury": 1,
#     "Hypertension": 1,
#     "SystolicBP": 160,
#     "DiastolicBP": 100,
#     "CholesterolTotal": 290,
#     "CholesterolLDL": 190,
#     "CholesterolHDL": 30,
#     "CholesterolTriglycerides": 350,
#     "FunctionalAssessment": 3,
#     "MemoryComplaints": 1,
#     "BehavioralProblems": 1,
#     "ADL": 2,
#     "Confusion": 1,
#     "Disorientation": 1,
#     "PersonalityChanges": 1,
#     "DifficultyCompletingTasks": 1,
#     "Forgetfulness": 1,
#     "Diagnosis": 1
# }

# # Fill any missing columns
# for col in all_features:
#     if col not in synthetic_patient:
#         synthetic_patient[col] = 0

# # Step 2: Predict baseline MMSE
# synthetic_df = pd.DataFrame([synthetic_patient])[all_features]
# synthetic_scaled = scaler.transform(synthetic_df)
# synthetic_mmse = rf_reg.predict(synthetic_scaled)[0]

# print("Initial MMSE Prediction (synthetic patient):", round(synthetic_mmse, 2))

# # Step 3: Run RL agent
# sample_dict = synthetic_patient.copy()

# env = MMSEEnv(model=xgb_reg, scaler=scaler, feature_names=all_features, initial_state=sample_dict)
# agent = QLearningAgent(actions=list(range(7)))  # 7 actions

# # RL training loop
# for episode in range(1000):
#     state = env.reset(sample_dict)
#     action = agent.choose_action(state)
#     next_state, reward, done = env.step(action)
#     agent.learn(state, action, reward, next_state)

# # Step 4: Simulate Top-K Actions
# state_key = agent.get_state_key(sample_dict)
# q_values = agent.q_table[state_key]
# top_actions = np.argsort(q_values)[::-1][:3]

# action_map = {
#     0: "Do Nothing", 1: "Improve Sleep", 2: "Increase Activity", 3: "Improve Diet",
#     4: "Reduce Alcohol", 5: "Manage BMI", 6: "Quit Smoking"
# }

# print("\nTop 3 Recommended Actions and Expected MMSE Impact:")
# for act in top_actions:
#     env.reset(sample_dict)
#     _, reward, _ = env.step(act)

#     post_df = pd.DataFrame([env.state])[all_features]
#     post_scaled = scaler.transform(post_df)
#     post_mmse = rf_reg.predict(post_scaled)[0]

#     print(f" -> {action_map[act]}:")
#     print(f"     Predicted MMSE: {round(post_mmse, 2)}")
#     print(f"     Change from baseline: {round(post_mmse - synthetic_mmse, 2)}")

# # Step 5: Full lifestyle optimization simulation
# optimized = sample_dict.copy()
# optimized.update({
#     "DietQuality": 9,
#     "BMI": 23,
#     "PhysicalActivity": 8,
#     "SleepQuality": 8,
#     "CholesterolLDL": 100,
#     "AlcoholConsumption": 0,
#     "Smoking": 0,
#     "Hypertension": 0,
#     "Diabetes": 0,
#     "Depression": 0,
#     "CardiovascularDisease": 0
# })

# opt_df = pd.DataFrame([optimized])[all_features]
# opt_scaled = scaler.transform(opt_df)
# opt_mmse = rf_reg.predict(opt_scaled)[0]

# print("\n--- Simulated Optimized Patient ---")
# print("Optimized MMSE Prediction:", round(opt_mmse, 2))
# print("Change from baseline:", round(opt_mmse - synthetic_mmse, 2))

# # Bonus: Show what the top 3 agent actions would do to optimized patient
# print("\nTop 3 Actions (Applied to Optimized Patient):")
# for act in top_actions:
#     env.reset(optimized)
#     _, _, _ = env.step(act)
    
#     post_df = pd.DataFrame([env.state])[all_features]
#     post_scaled = scaler.transform(post_df)
#     post_mmse = rf_reg.predict(post_scaled)[0]

#     print(f" -> {action_map[act]}:")
#     print(f"     Predicted MMSE: {round(post_mmse, 2)}")
#     print(f"     Change from optimized: {round(post_mmse - opt_mmse, 2)}")
