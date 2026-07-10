import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.fraud_detection.logger import logging
from src.fraud_detection.exception import CustomException
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.fraud_detection.utils.utils import save_object, evaluate_model


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','Model.joblib')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            target_column = "isFraud"

            X_train = train_array.drop(columns=[target_column])
            y_train = train_array[target_column]

            X_test = test_array.drop(columns=[target_column])
            y_test = test_array[target_column]
            logging.info("Independent and dependent variables split successfully")

            # For XGBoost
            scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            logging.info(f"Scale_Pos_Weight: {scale_pos_weight}")

            models = {
                'Random Forest Classifier':RandomForestClassifier(
                    n_estimators=200, max_depth=12, class_weight="balanced",n_jobs=-1, random_state=42),
                'XG Boost':XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    gamma=0.1,
                    tree_method="hist",
                    eval_metric="aucpr",
                    n_jobs=-1,
                    random_state=42),
                'Cat Boost':CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    auto_class_weights="Balanced",
                    eval_metric="PRAUC",
                    random_state=42,
                    loss_function="Logloss",
                    verbose=False)
                }
            
            model_report, trained_models = evaluate_model(X_train, y_train, X_test, y_test, models)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["f1_score"]
            )

            best_model_metrics = model_report[best_model_name]

            best_model = trained_models[best_model_name]

            print(
                f"Best Model Found, Model Name: {best_model_name}, "
                f"F1 Score: {best_model_metrics['f1_score']:.4f}"
            )         
            print('\n====================================================================================\n')

            logging.info(
                f"""
            Best Model : {best_model_name}
            Accuracy   : {best_model_metrics['accuracy']:.4f}
            Precision  : {best_model_metrics['precision']:.4f}
            Recall     : {best_model_metrics['recall']:.4f}
            F1 Score   : {best_model_metrics['f1_score']:.4f}
            ROC-AUC    : {best_model_metrics['roc_auc']}
            PR-AUC     : {best_model_metrics['pr_auc']}
            """
            )        

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          
        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e,sys)