import os
import sys
import mlflow
from src.fraud_detection.exception import CustomException
from src.fraud_detection.logger import logging 
import mlflow.sklearn
from urllib.parse import urlparse
from src.fraud_detection.utils.utils import load_object
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred, pred_proba=None):
        accuracy = accuracy_score(actual, pred)

        precision = precision_score(
            actual,
            pred,
            zero_division=0
        )

        recall = recall_score(
            actual,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            pred,
            zero_division=0
        )

        roc_auc = None
        pr_auc = None

        if pred_proba is not None:
            roc_auc = roc_auc_score(actual, pred_proba)
            pr_auc = average_precision_score(actual, pred_proba)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }
    

    def initiate_model_evaluation(self, test_array):
        try:
            target_column = "isFraud"

            X_test = test_array.drop(columns=[target_column])
            y_test = test_array[target_column]

            model_path = os.path.join("artifacts", "Model.joblib")

            model = load_object(model_path)


            # mlflow.set_registry_uri(" ")
                        
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)
            logging.info("Loading trained model...")
            logging.info("Evaluating model on test dataset...")

            with mlflow.start_run():

                y_pred = model.predict(X_test)

                y_pred_prob = None
                if hasattr(model, "predict_proba"):
                    y_pred_prob = model.predict_proba(X_test)[:, 1]

                metrics = self.eval_metrics(
                    y_test,
                    y_pred,
                    y_pred_prob
                )

                mlflow.log_metric("Accuracy Score", metrics["accuracy"])
                mlflow.log_metric("Precision Score", metrics["precision"])
                mlflow.log_metric("Recall Score", metrics["recall"])
                mlflow.log_metric("F1 Score", metrics["f1_score"])

                if metrics["roc_auc"] is not None:
                    mlflow.log_metric("ROC-AUC Score", metrics["roc_auc"])

                if metrics["pr_auc"] is not None:
                    mlflow.log_metric("PR-AUC Score", metrics["pr_auc"])

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "Model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "Model")
                
        except Exception as e:
            logging.info("Exception occurred during model evaluation")
            raise CustomException(e, sys)