import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import pickle
import os 

from imblearn.over_sampling import SMOTE

def rf_objective(trial, X_train, y_train, X_test, y_test,return_model=False):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", [ "sqrt", "log2"]),
        "random_state": 42 
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
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
        "random_state": 42, 
        "n_jobs": -1 
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
        "random_state": 42, 
        "use_label_encoder": False, 
        "eval_metric": 'logloss' 
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
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
    

    if solver == "liblinear":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    elif solver == "saga":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]) 
    else: 
        penalty = "l2"

    params = {
        "C": trial.suggest_float('C', 0.001, 10, log=True),
        "solver": solver,
        "penalty": penalty,
        "random_state": 42, 
        "max_iter": 2000 
    }
    
    if penalty == "elasticnet" and solver == "saga":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    try:
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auc_pr = auc(recall, precision)
        if return_model:
            return model
        return auc_pr
    except ValueError: 
        return 0.0 
    except Exception: 
        return 0.0


def lda_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    solver = trial.suggest_categorical("solver", ["lsqr", "eigen"]) 
    params = {"solver": solver}
    if solver in ["lsqr", "eigen"]: 
        params["shrinkage"] = trial.suggest_uniform('shrinkage', 0.0, 1.0)
    
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
        "n_jobs": -1 
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
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42 
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
    params = {
        'var_smoothing': trial.suggest_loguniform('var_smoothing', 1e-10, 1e-3) 
    }
    model = GaussianNB(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr

def svm_objective(trial, X_train, y_train, X_test, y_test, return_model=False):
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    params = {
        "C": trial.suggest_loguniform("C", 1e-4, 10.0),
        "kernel": kernel,
        "random_state": 42 
    }
    if kernel in ["rbf", "poly", "sigmoid"]:
        
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"]) 
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5) 
        
    model = SVC(probability=True, **params) 
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    if return_model:
        return model
    return auc_pr



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

def evaluate_model(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    precision_score_val = precision_score(y_true, y_pred, zero_division=0) 
    recall_score_val = recall_score(y_true, y_pred, zero_division=0) 
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4: 
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else: 
        tn, fp, fn, tp = 0,0,0,0 
        specificity = 0
        
        

    sensitivity = recall_score_val
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'precision': precision_score_val,
        'recall': recall_score_val,
        'specificity': specificity,
        'sensitivity': sensitivity
    }


def optimize_models(models_dict, X_train_orig, y_train_orig, X_test_orig, y_test_orig, csv_filename, n_trials=50):
    
    X_train_clipped = np.clip(X_train_orig.astype(float), a_min=-1e10, a_max=1e10) 
    X_test_clipped = np.clip(X_test_orig.astype(float), a_min=-1e10, a_max=1e10)   

    
    print(f"Original training data shape: {X_train_clipped.shape}")
    print(f"Class distribution before SMOTE: \n{pd.Series(y_train_orig).value_counts(normalize=True)}")
    
    smote = SMOTE(random_state=1) 
    X_train_smote, y_train_smote = smote.fit_resample(X_train_clipped, y_train_orig)
    
    print(f"SMOTE'd training data shape: {X_train_smote.shape}")
    print(f"Class distribution after SMOTE: \n{pd.Series(y_train_smote).value_counts(normalize=True)}")

    
    base_dir = os.path.splitext(os.path.basename(csv_filename))[0] + "_smote_results" 
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    all_metrics = []
    for name, objective_func in models_dict.items():
        print(f"\nOptimizing {name} with SMOTE'd training data...")
        study = optuna.create_study(direction="maximize")
        try:
            
            study.optimize(lambda trial: objective_func(trial, X_train_smote, y_train_smote, X_test_clipped, y_test_orig), n_trials=n_trials)
        except Exception as e:
            print(f"Error during optimization for {name}: {e}")
            
            if "Singular matrix" in str(e) or "collinear" in str(e): 
                print("This might be due to perfect multicollinearity. Skipping this model.")
            continue 
            
        
        best_params = study.best_params
        best_model = objective_func(optuna.trial.FixedTrial(best_params), X_train_smote, y_train_smote, X_test_clipped, y_test_orig, return_model=True)
        
        
        model_filename = os.path.join(base_dir, f'{name}_tuned_smote.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        
        
        params_filename = os.path.join(base_dir, f'{name}_best_params_smote.txt')
        with open(params_filename, 'w') as f:
            f.write(f"Best parameters for {name} (trained with SMOTE):\n")
            for param_name, value in best_params.items():
                f.write(f"{param_name}: {value}\n")
        
        
        y_train_pred = best_model.predict(X_train_clipped)
        y_train_prob = best_model.predict_proba(X_train_clipped)[:, 1]
        train_metrics = evaluate_model(y_train_orig, y_train_pred, y_train_prob)

        
        y_test_pred  = best_model.predict(X_test_clipped)
        y_test_prob  = best_model.predict_proba(X_test_clipped)[:, 1]
        test_metrics = evaluate_model(y_test_orig, y_test_pred, y_test_prob)

        rec = {'model': name}
        for k, v in test_metrics.items():
            rec[f"test_{k}"] = v
        for k, v in train_metrics.items():
            rec[f"train_{k}"] = v
        rec.update(best_params)
        all_metrics.append(rec)
        
        print(f"[{name} with SMOTE] train_auc_pr={train_metrics['auc_pr']:.4f}  test_auc_pr={test_metrics['auc_pr']:.4f}")
    
    
    metrics_filename = os.path.join(base_dir, 'all_tuned_model_metrics_smote.csv')
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_filename, index=False)

    print(f"\nSMOTE results saved in directory: {base_dir}")
    return metrics_df


def load_data(csv_file):
    data = pd.read_csv(csv_file)
    
    if 'TASTE' not in data.columns:
        raise ValueError("Column 'TASTE' not found in the CSV file.")
    X = data.drop('TASTE', axis=1)
    y = data['TASTE']
    
    y = y.astype(int) 
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

def main(csv_file_path, results_base_name):
    X_train, X_test, y_train, y_test = load_data(csv_file_path)
    optimize_models(models, X_train, y_train, X_test, y_test, results_base_name, n_trials=50) 
    

if __name__ == "__main__":

    main("/home/pavit21178/BTP/new_data_molecules/encodings_new/Morgan_fingerprints.csv","SMOTEMorganFingerprntsResults")
    