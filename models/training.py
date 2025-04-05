import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, classification_report

# Training the data against both models to which performs better
def train_models(Xc_train, yc_train, Xr_train, yr_train):
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    xgb_clf = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='logloss')

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    xgb_reg = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)

    rf_clf.fit(Xc_train, yc_train)
    xgb_clf.fit(Xc_train, yc_train)
    rf_reg.fit(Xr_train, yr_train)
    xgb_reg.fit(Xr_train, yr_train)

    return rf_clf, xgb_clf, rf_reg, xgb_reg

# Evaluating the models
def evaluate_models(rf_clf, xgb_clf, rf_reg, xgb_reg, Xc_test, yc_test, Xr_test, yr_test):
    print("\n--- Classification Results ---")
    for name, model in zip(["Random Forest", "XGBoost"], [rf_clf, xgb_clf]):
        preds = model.predict(Xc_test)
        print(f"{name} Accuracy: {accuracy_score(yc_test, preds):.2f}")
        print(f"{name} AUC: {roc_auc_score(yc_test, preds):.2f}")
        print(classification_report(yc_test, preds))

    print("\n--- Regression Results (MMSE) ---")
    for name, model in zip(["Random Forest", "XGBoost"], [rf_reg, xgb_reg]):
        preds = model.predict(Xr_test)
        print(f"{name} MAE: {mean_absolute_error(yr_test, preds):.2f}")

# Showing the feature importance
def show_feature_importance(model, X):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=True).plot(kind='barh', figsize=(10, 8), title='Feature Importance')
    plt.tight_layout()
    plt.show()

# Showing the SHAP explanation
def shap_explanation(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)