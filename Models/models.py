import pickle

import pandas as pd
import numpy as np

from sklearn import clone
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

path = 'D:/Users/nickt/Credit_Card_Project'

###############################################################
# Load in Data 
###############################################################

df = pd.read_csv(f'{path}/Data/CreditCardData_cleaned.csv')
df = df.dropna().reset_index(drop=True)

###############################################################
# Define variables 
###############################################################

y = df['Fraud']
X = df.drop(columns=['Fraud'])

###############################################################
# Logistic Regression and KNN 
###############################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

logreg = LogisticRegression(random_state=42, max_iter=10000, )
knn = KNeighborsClassifier(n_jobs=-1)

# Correct hyperparameters for each model
param_grids = {
    'Logistic Regression': {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2", "l1"],
        "solver": ["saga"], 
        "class_weight": [None, "balanced", {0: 1, 1: 3}, {0: 1, 1: 5}]    
    },
    'KNN': {
        "n_neighbors": [3, 9, 15, 25],
        "weights": ["uniform", "distance"], 
        "metric": ['euclidean']
    }
}

# Names now match the actual models
models = {'Logistic Regression': logreg, 'KNN': knn}

# Proper cross-validation splitter (stratified, since fraud is imbalanced)
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = {}
results_summary = []
evaluation_results = []
feature_importance_values = {}
feature_names = X.columns.tolist()

for model_name, model in models.items():
    print(f"Starting Grid Search for {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=cv_splitter,
        scoring='average_precision',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit on the training set only (not the full X, y) to avoid leakage into the held-out test set
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model

    if model_name == 'Logistic Regression':
        importances = np.abs(best_model.coef_[0])
    else:
        X_perm_sample, y_perm_sample = resample(X_train_scaled, y_train, n_samples=5000, stratify=y_train, random_state=42)
        perm_result = permutation_importance(best_model, X_perm_sample, y_perm_sample, n_repeats=5, random_state=42, scoring='f1', n_jobs=-1)
        importances = perm_result.importances_mean

    feature_importance_values[model_name] = importances

    results_summary.append({
        'model': model_name,
        'best_params': grid_search.best_params_,
        'best_cv_f1': grid_search.best_score_
    })

    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = best_model.decision_function(X_test_scaled)

    y_pred = (y_proba >= 0.5).astype(int)

    evaluation_results.append({
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
        'y_true': y_test.to_numpy(),
        'y_pred': y_pred,
        'y_proba': y_proba,
    })

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation average_precision: {grid_search.best_score_:.3f}\n")

###############################################################
# Export the models and results to a pickle file
###############################################################

summary_df = pd.DataFrame(results_summary)
feature_importance_df = pd.DataFrame([
    {'model': model_name, 'feature': feature, 'importance': importance}
    for model_name, importances in feature_importance_values.items()
    for feature, importance in zip(feature_names, importances)
])

output_path = f"{path}/Models/trained_models.pkl"

with open(output_path, 'wb') as f:
    pickle.dump({
        'best_models': best_models,
        'summary_df': summary_df,
        'evaluation_results': evaluation_results,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'feature_importance_values': feature_importance_values,
        'feature_importance_df': feature_importance_df
    }, f)