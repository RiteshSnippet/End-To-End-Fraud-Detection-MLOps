import os
import sys
import joblib
import numpy as np
import pandas as pd
from src.fraud_detection.logger import logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from src.fraud_detection.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)

    except Exception as e:
        logging.info("Exception occurred in save_object function utils")
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            y_test_prob = None
            if hasattr(model, "predict_proba"):
                y_test_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(
                y_test,
                y_test_pred
            )

            precision = precision_score(
                y_test,
                y_test_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_test_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_test_pred,
                zero_division=0
            )

            roc_auc = None
            pr_auc = None

            if y_test_prob is not None:
                roc_auc = roc_auc_score(
                    y_test,
                    y_test_prob
                )

                pr_auc = average_precision_score(
                    y_test,
                    y_test_prob
                )

            report[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            }

            trained_models[model_name] = model

        return report, trained_models

    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        return joblib.load(file_path)

    except Exception as e:
        logging.info("Exception occurred in load_object function utils")
        raise CustomException(e, sys)