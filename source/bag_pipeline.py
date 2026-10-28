import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle

#################################################################
#              PERFORMANCE EVALUATION UTILS
#################################################################

def rmse(y_true, y_pred):
    """Root Mean Squared Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Results
    -------
    rsme : np.float
        Root mean squared error value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def pearson_r(y_true, y_pred):
    """Correlazione di Pearson tra y_true e y_pred.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Results
    -------
    p : np.float
        Pearson's correlation coefficient
    """
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]

def summarize(split, metrics):
    """
    Summary of performance evaluation for train/test set.
    
    Parameters
    ----------
    split : str
        'train' or 'test'
    metrics: dict
        Performance evaluation dictionary (keys : set kind, values : performance metrics)
    
    Results
    -------
    summary : dict
        Summary of performance evaluation for train/test set (keys : metric, values : (mean, std))
    """
    return {
        m: (np.mean(vals), np.std(vals))
        for m, vals in metrics[split].items()
        }

#################################################################
#              COVARIANCE RESIDUALIZATION
#################################################################

class ResidualizeCovarites(BaseEstimator, TransformerMixin):
    """
    Residualize features w.r.t. covariates (e.g., sex, education).
    Covariates are assumed to be the LAST n_cov columns of X.
    """
    def __init__(self, n_cov=2):
        self.n_cov = n_cov

    def fit(self, X, y=None):
        covariates = X[:, -self.n_cov:]
        features   = X[:, :-self.n_cov]
        self.models_ = [
            LinearRegression().fit(covariates, features[:, i])
            for i in range(features.shape[1])
        ]
        return self

    def transform(self, X):
        covariates = X[:, -self.n_cov:]
        features   = X[:, :-self.n_cov]
        return np.column_stack([
            features[:, i] - self.models_[i].predict(covariates)
            for i in range(features.shape[1])
        ])  # covariates are dropped after residualization
    
#################################################################
#              NESTED CROSS-VALIDATION PIPELINE
#################################################################

def nested_cv_predictions(
    X,
    y,
    strat_labels,
    pipeline,
    n_iter = 20,
    scoring = 'neg_mean_absolute_error',
    n_splits=5,
    groups_labels = None,
):
    """
    Nested Cross-Validation Scheme to predict the chronological age from the features

    Parameters
    ----------
    X : np.ndarray (n_subjects, n_features)
        Features set
    y : np.ndarray (n_subjects,)
        Target variable (age)
    strat_labels : np.ndarray (n_subjects,)
        Quantiles stratification for CV balance
    pipeline : sklearn.pipeline.Pipeline
        Regression pipeline
    n_iter : int (default = 20)
        Number of iteration for the RandomizedarchCV for hyperparameters optimization
    scoring : str (default = 'neg_mean_absolute_error')
        Scoring method ('r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', etc...)
    n_splits : int (default = 5)
        Number of splits for CV
    groups_labels : np.ndarray (n_subjects,)
        Group labels for CV stratification
        
    Results
    -------
    y_pred : np.ndarray
        Predicted values
    metrics: dict
        Performance evaluation dictionary (keys : set kind, values : performance metrics)
    train_sum : dict
        Summary of performance evaluation for train set (keys : metric, values : (mean, std))
    test_sum : dict
        Summary of performance evaluation for test set (keys : metric, values : (mean, std))
    """
    if groups_labels is not None:
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize results
    y_pred = np.zeros_like(y, dtype=float)
    metrics = {
        "train": {"mae": [], "rmse": [], "r2": [], "p": []},
        "test":  {"mae": [], "rmse": [], "r2": [], "p": []},
        "null": {"mae": [], "rmse": [], "r2": [], "p": []}}
    # Initialize hyperparameter search
    if isinstance(pipeline['regressor'], SVR):
        param_dist = {
                    "regressor__C": np.linspace(0.1, 1000, 20),
                    "regressor__degree": list(range(1,4)),
                }
    else:
        param_dist = {"regressor__alpha": np.logspace(-2, 5, 100)}

    # Initialize best parameters dictionary
    best_parameters = {}

    # Nested CV scheme
    split = outer_cv.split(X, strat_labels, groups_labels) if groups_labels is not None else outer_cv.split(X, strat_labels)
    for fold_idx, (train_idx, test_idx) in enumerate(split, start=1):

        # Train and test sets split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        strat_train = strat_labels[train_idx]
        groups_train = groups_labels[train_idx] if groups_labels is not None else None

        inner_split = inner_cv.split(X_train, strat_train, groups_train) if groups_labels is not None else inner_cv.split(X_train, strat_train)
        # Hyperparameter optimization
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter, 
            cv=inner_split,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        # Select hyperparameters to have the best model
        search.fit(X_train, y_train) 
        best_model = search.best_estimator_ 

        # Save best parameters
        best_parameters[fold_idx] = search.best_params_

        # Prediction for best model
        y_train_pred = best_model.predict(X_train) 
        y_test_pred  = best_model.predict(X_test) 

        # Save predictions
        y_pred[test_idx] = y_test_pred 

        # Save accuracy train e test
        for split_name, yt, yp in [("train", y_train, y_train_pred),("test",  y_test,  y_test_pred)]:
            metrics[split_name]["mae"].append(mean_absolute_error(yt, yp))
            metrics[split_name]["rmse"].append(rmse(yt, yp))
            metrics[split_name]["r2"].append(r2_score(yt, yp))
            metrics[split_name]["p"].append(pearson_r(yt, yp))

    train_sum = summarize("train", metrics)
    test_sum  = summarize("test", metrics)
    
    return y_pred, metrics, train_sum, test_sum, best_parameters, best_model

def save_performance(performance_path, best_model, best_parameters, metrics, train_sum, test_sum, residual_gap, predicted_age, corrected_age, demo):
    """
    Save predictions and performances

    Parameters
    ----------
    performance_path : str
        '.txt' format path to save results
    metrics: dict
        Performance evaluation dictionary (keys : set kind, values : performance metrics)
    train_sum : dict
        Summary of performance evaluation for train set (keys : metric, values : (mean, std))
    test_sum : dict
        Summary of performance evaluation for test set (keys : metric, values : (mean, std))
    residual_gap : np.ndarray (n_subjects,)
        Brain Age Gap
    predicted_age : np.ndarray (n_subjects,)
        Predicted values
    corrected_age : np.ndarray (n_subjects,)
        Age-bias corrected predicted values
    demo : pd.DataFrame
        Demographics info
    """
    summary_path = performance_path.replace(".txt", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MODEL SUMMARY (Nested CV con StratifiedGroupKFold)\n\n")

        # per-fold metrics
        n_folds = len(metrics["train"]["mae"])
        for i in range(n_folds):
            f.write(f"Fold N: {i+1}\n")
            f.write(
                "Train: MAE {:.6f}, RMSE {:.6f}, R2 {:.6f}, p {:.6f}\n".format(
                    metrics["train"]["mae"][i],
                    metrics["train"]["rmse"][i],
                    metrics["train"]["r2"][i],
                    metrics["train"]["p"][i],
                )
            )
            f.write(
                "Test: MAE {:.6f}, RMSE {:.6f}, R2 {:.6f}, p {:.6f}\n".format(
                    metrics["test"]["mae"][i],
                    metrics["test"]["rmse"][i],
                    metrics["test"]["r2"][i],
                    metrics["test"]["p"][i],
                )
            )

        # summary metrics
        f.write("Summary metrics over all CV splits\n")
        f.write(
            "Train: MAE {:.6f} ± {:.6f}, RMSE {:.6f} ± {:.6f}, "
            "R2 {:.6f} ± {:.6f}, p {:.6f} ± {:.6f}\n".format(
                train_sum["mae"][0],  train_sum["mae"][1],
                train_sum["rmse"][0], train_sum["rmse"][1],
                train_sum["r2"][0],   train_sum["r2"][1],
                train_sum["p"][0],    train_sum["p"][1],
            )
        )
        f.write(
            "Test: MAE {:.6f} ± {:.6f}, RMSE {:.6f} ± {:.6f}, "
            "R2 {:.6f} ± {:.6f}, p {:.6f} ± {:.6f}\n".format(
                test_sum["mae"][0],  test_sum["mae"][1],
                test_sum["rmse"][0], test_sum["rmse"][1],
                test_sum["r2"][0],   test_sum["r2"][1],
                test_sum["p"][0],    test_sum["p"][1],
            )
        )

    # TABELLA outcomes con results xsubject

    results_df = pd.DataFrame({
        "id": demo['id'].values,
        "age": demo['age'].values,
        "predicted_age": predicted_age,
        "corrected_age": corrected_age,
        "bag": residual_gap
    })

    output_path = performance_path.replace(".txt", ".csv")
    results_df.to_csv(output_path, index=False)


    metrics_path = performance_path.replace('.txt', '_metrics.pickle')
    with open(metrics_path, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    best_params_path = performance_path.replace('.txt', '_best_parameters.pickle')
    with open(best_params_path, 'wb') as handle:
        pickle.dump(best_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            
    best_model_path = performance_path.replace('.txt', '_best_model.pickle')
    with open(best_model_path, 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
