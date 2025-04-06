import numpy as np
import pandas as pd
import random
from collections import defaultdict

# class MMSEEnv:
#     def __init__(self, model, scaler, feature_names, initial_state):
#         self.model = model
#         self.scaler = scaler
#         self.feature_names = feature_names
#         self.state = initial_state.copy()
#         self.done = False

#     def reset(self, new_state):
#         self.state = new_state.copy()
#         self.done = False
#         return self.state

#     def step(self, action):
#         next_state = self.state.copy()

#         # Define action impact
#         if action == 1:
#             next_state['SleepQuality'] = min(10, next_state['SleepQuality'] + 1)
#         elif action == 2:
#             next_state['PhysicalActivity'] = min(10, next_state['PhysicalActivity'] + 1)
#         elif action == 3:
#             next_state['DietQuality'] = min(10, next_state['DietQuality'] + 1)
#         elif action == 4:
#             next_state['AlcoholConsumption'] = max(0, next_state['AlcoholConsumption'] - 2)
#         elif action == 5:
#             if next_state['BMI'] > 24:
#                 next_state['BMI'] -= 1
#         elif action == 6:
#             next_state['Smoking'] = 0

#         input_df = pd.DataFrame([next_state])[self.feature_names]
#         input_scaled = self.scaler.transform(input_df)
#         next_mmse = self.model.predict(input_scaled)[0]

#         old_input_df = pd.DataFrame([self.state])[self.feature_names]
#         old_scaled = self.scaler.transform(old_input_df)
#         old_mmse = self.model.predict(old_scaled)[0]

#         reward = next_mmse - old_mmse

#         # Reward modifiers based on patient conditions
#         if self.state.get('Hypertension') == 1 and action in [2, 3]:
#             reward *= 1.2
#         if self.state.get('CardiovascularDisease') == 1 and action in [2, 3]:
#             reward *= 1.2
#         if self.state.get('Diabetes') == 1 and action in [3, 5]:
#             reward *= 1.2
#         if self.state.get('Depression') == 1 and action in [1, 2]:
#             reward *= 1.2
#         if self.state.get('CholesterolLDL', 0) > 130 and action in [2, 3]:
#             reward *= 1.3

#         self.state = next_state
#         self.done = True
#         return next_state, reward, self.done


# class QLearningAgent:
#     def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
#         self.q_table = defaultdict(lambda: np.zeros(len(actions)))
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.actions = actions

#     def get_state_key(self, state):
#         return tuple(round(state[k], 1) for k in sorted(state.keys()))

#     def choose_action(self, state):
#         state_key = self.get_state_key(state)
#         if random.random() < self.epsilon:
#             return random.choice(self.actions)
#         return np.argmax(self.q_table[state_key])

#     def learn(self, state, action, reward, next_state):
#         state_key = self.get_state_key(state)
#         next_key = self.get_state_key(next_state)
#         predict = self.q_table[state_key][action]
#         target = reward + self.gamma * np.max(self.q_table[next_key])
#         self.q_table[state_key][action] += self.alpha * (target - predict)


class MMSEEnv:
    def __init__(self, model, scaler, feature_names, initial_state, max_steps=5):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.initial_state = initial_state.copy()
        self.state = self.initial_state.copy()
        self.max_steps = max_steps
        self.turn = 0
        self.initial_mmse = self._get_mmse(self.initial_state)

    def _get_mmse(self, state):
        df = pd.DataFrame([state])[self.feature_names]
        scaled = self.scaler.transform(df)
        return self.model.predict(scaled)[0]

    def reset(self, new_state):
        self.initial_state = new_state.copy()
        self.state = new_state.copy()
        self.turn = 0
        self.initial_mmse = self._get_mmse(self.initial_state)
        return self.state

    def step(self, action):
        self.turn += 1
        next_state = self.state.copy()

        # Define action impact
        if action == 1:
            next_state['SleepQuality'] = min(10, next_state['SleepQuality'] + 1)
        elif action == 2:
            next_state['PhysicalActivity'] = min(10, next_state['PhysicalActivity'] + 1)
        elif action == 3:
            next_state['DietQuality'] = min(10, next_state['DietQuality'] + 1)
        elif action == 4:
            next_state['AlcoholConsumption'] = max(0, next_state['AlcoholConsumption'] - 2)
        elif action == 5:
            if next_state['BMI'] > 24:
                next_state['BMI'] -= 1
        elif action == 6:
            next_state['Smoking'] = 0

        self.state = next_state

        done = self.turn >= self.max_steps
        final_mmse = self._get_mmse(self.state)
        reward = (final_mmse - self.initial_mmse) * 10  # Amplify reward

        # Reward modifiers based on patient conditions
        if self.state.get('Hypertension') == 1 and action in [2, 3]:
            reward *= 1.2
        if self.state.get('CardiovascularDisease') == 1 and action in [2, 3]:
            reward *= 1.2
        if self.state.get('Diabetes') == 1 and action in [3, 5]:
            reward *= 1.2
        if self.state.get('Depression') == 1 and action in [1, 2]:
            reward *= 1.2
        if self.state.get('CholesterolLDL', 0) > 130 and action in [2, 3]:
            reward *= 1.3

        return next_state, reward, done

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = actions

    def get_state_key(self, state):
        return tuple(round(state[k], 1) if isinstance(state[k], (int, float)) else str(state[k]) for k in sorted(state.keys()))

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        predict = self.q_table[state_key][action]
        target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[state_key][action] += self.alpha * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)