import os
import logging
import pickle
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, f1_score, precision_score,
    recall_score, confusion_matrix, auc
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Logging configuration\LOG_DIR = "logs"
LOG_DIR="/home/pavit21178/BTP/redoing_work/pipeline_work/pipelines/logs_balanced"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define objective functions for each model

def rf_objective(trial, X_train, y_train, X_test, y_test,return_model=False):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", [ "sqrt", "log2"]),
        "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    accuracy = accuracy_score(y_test, y_pred)
    if return_model:
        return model
    return auc_pr

def lgbm_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def xgb_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def lr_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "C": trial.suggest_float('C', 0.001, 10, log=True),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
    }
    try:
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auc_pr = auc(recall, precision)
        if return_model:
            return model
        return auc_pr
    except:
        return 0

def lda_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "solver": trial.suggest_categorical("solver", ["lsqr", "eigen"]),
        "shrinkage": trial.suggest_uniform('shrinkage', 0.0, 1.0)
    }
    model = LDA(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def qda_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "reg_param": trial.suggest_uniform("reg_param", 0.0, 1.0),
    }
    model = QDA(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def knn_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 5),
    }
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def extra_trees_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", [ "sqrt", "log2"]),
    }
    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def nb_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {}  # No hyperparameters to tune for Naive Bayes in this context
    model = GaussianNB(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def svm_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    params = {
        "C": trial.suggest_loguniform("C", 1e-4, 10.0),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale"]),
    }
    model = SVC(probability=True, **params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

# Define models dictionary with their corresponding objective functions
models = {
    "RandomForest": rf_objective,
    "LightGBM": lgbm_objective,
    "XGBoost": xgb_objective,
    "LogisticRegression": lr_objective,
    "LinearDiscriminantAnalysis": lda_objective,
    "QuadraticDiscriminantAnalysis": qda_objective,
    "KNeighbors": knn_objective,
    "ExtraTrees": extra_trees_objective,
    "NaiveBayes": nb_objective,
    "SVM": svm_objective
}
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, auc


import optuna
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
import json

# Evaluation utility
def evaluate_model(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'precision': precision_val,
        'recall': recall_val,
        'specificity': specificity
    }

# General pipeline including training metrics
def run_general(csv_file, output_dir, n_trials=150):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    X = df.drop('TASTE', axis=1)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    X = np.clip(X, -1e+38, 1e+38)
    y = df['TASTE']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

        # ── APPLY SMOTE ──────────────────────────────────────────────────────
    sm = SMOTE(random_state=1)
    X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
    logger.info(f"After SMOTE, training set size: {X_train_scaled.shape[0]} samples")

    metrics_list = []
    for name, obj in models.items():  # add other models
        logger.info(f"Optimizing {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: obj(trial, X_train_scaled, y_train, X_test_scaled, y_test),
            n_trials=n_trials
        )
        best_params = study.best_params
        logger.info(f"Best params for {name}: {best_params}")

        model = obj(
            optuna.trial.FixedTrial(best_params),
            X_train_scaled, y_train, X_test_scaled, y_test,
            return_model=True
        )
        # Save model
        with open(os.path.join(output_dir, f"{name}.pkl"), 'wb') as f:
            pickle.dump(model, f)
        params_path = os.path.join(output_dir, f"{name}_best_params.json")
        with open(params_path, "w") as fp:
            json.dump(best_params, fp, indent=4)
        logger.info(f"Wrote best parameters for {name} to {params_path}")
        # Evaluate on test
        y_test_pred = model.predict(X_test_scaled)
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)

        # Evaluate on train
        y_train_pred = model.predict(X_train_scaled)
        y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
        train_metrics = evaluate_model(y_train, y_train_pred, y_train_prob)

        # Combine
        record = {'model': name}
        # prefix metrics
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        metrics_list.append(record)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(output_dir, 'general_metrics.csv'), index=False)
    logger.info("General pipeline completed with train and test metrics.")

# CV pipeline including training metrics
def run_cv(csv_file, output_dir, n_splits=5, n_trials=100):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    X = df.drop('TASTE', axis=1)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    X = np.clip(X, -1e+38, 1e+38)
    y = df['TASTE']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    cv_records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        logger.info(f"Fold {fold}/{n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        sm = SMOTE(random_state=1)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
        logger.info(f"After SMOTE, training set size: {X_train_scaled.shape[0]} samples")

        for name, obj in models.items():  # add others
            logger.info(f"Optimizing {name} on fold {fold}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: obj(trial, X_train_scaled, y_train, X_test_scaled, y_test),
                n_trials=n_trials
            )
            best_params = study.best_params

            model = obj(
                optuna.trial.FixedTrial(best_params),
                X_train_scaled, y_train, X_test_scaled, y_test,
                return_model=True
            )
            params_path = os.path.join(output_dir, f"fold{fold}_{name}_best_params.json")
            with open(params_path, "w") as fp:
                json.dump(best_params, fp, indent=4)
            logger.info(f"Wrote best parameters for {name} to {params_path}")
            # Save model
            with open(os.path.join(output_dir, f"fold{fold}_{name}.pkl"), 'wb') as f:
                pickle.dump(model, f)

            # Test eval
            y_test_pred = model.predict(X_test_scaled)
            y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
            test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)

            # Train eval
            y_train_pred = model.predict(X_train_scaled)
            y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
            train_metrics = evaluate_model(y_train, y_train_pred, y_train_prob)

            rec = {'fold': fold, 'model': name}
            rec.update({f"test_{k}": v for k, v in test_metrics.items()})
            rec.update({f"train_{k}": v for k, v in train_metrics.items()})
            cv_records.append(rec)

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(os.path.join(output_dir, 'cv_metrics.csv'), index=False)
    logger.info("Cross-validation pipeline completed with train and test metrics.")

# Entry point
if __name__ == "__main__":
    # run_general(csv_file="/home/pavit21178/BTP/redoing_work/pipeline_work/datasets of 3 encodings/mol2vec data/peptides_with_mol2vec.csv", output_dir="/home/pavit21178/BTP/redoing_work/pipeline_work/pipelines/balanced/general_mol2vec/peptides", n_trials=150)
    # run_general(csv_file="/home/pavit21178/BTP/redoing_work/pipeline_work/datasets of 3 encodings/mol2vec data/molecules_with_mol2vec.csv", output_dir="/home/pavit21178/BTP/redoing_work/pipeline_work/pipelines/balanced/general_mol2vec/molecules", n_trials=150)
    run_general(csv_file="/home/pavit21178/BTP/redoing_work/pipeline_work/datasets of 3 encodings/mol2vec data/combined_with_mol2vec.csv", output_dir="/home/pavit21178/BTP/redoing_work/pipeline_work/pipelines/balanced/general_mol2vec/combined", n_trials=150)
    

    


