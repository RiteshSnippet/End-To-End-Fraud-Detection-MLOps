import os
import sys
import pandas as pd
from src.fraud_detection.logger import logging
from src.fraud_detection.utils.utils import load_object
from src.fraud_detection.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass


    def predict(self, features):
        try:
            preprocessor_path = os.path.join(
                "artifacts",
                "Preprocessor.joblib"
            )
            model_path = os.path.join(
                "artifacts",
                "Model.joblib"
            )

            preprocessor = load_object(
                preprocessor_path
            )
            model = load_object(
                model_path
            )

            transformed_data = preprocessor.transform(
                features
            )
            prediction = model.predict(
                transformed_data
            )

            prediction_probability = None
            if hasattr(model, "predict_proba"):
                prediction_probability = model.predict_proba(
                    transformed_data
                )[:, 1]

            return prediction, prediction_probability

        except Exception as e:
            logging.info("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        amount: float,
        oldbalanceOrg: float,
        newbalanceOrig: float,
        oldbalanceDest: float,
        newbalanceDest: float,
        errorBalanceOrig: float,
        errorBalanceDest: float,
        hour: int,
        day: int,
        type: str
    ):

        self.amount = amount
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig
        self.oldbalanceDest = oldbalanceDest
        self.newbalanceDest = newbalanceDest
        self.errorBalanceOrig = errorBalanceOrig
        self.errorBalanceDest = errorBalanceDest
        self.hour = hour
        self.day = day
        self.type = type

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {

                "amount": [self.amount],

                "oldbalanceOrg": [
                    self.oldbalanceOrg
                ],

                "newbalanceOrig": [
                    self.newbalanceOrig
                ],

                "oldbalanceDest": [
                    self.oldbalanceDest
                ],

                "newbalanceDest": [
                    self.newbalanceDest
                ],

                "errorBalanceOrig": [
                    self.errorBalanceOrig
                ],

                "errorBalanceDest": [
                    self.errorBalanceDest
                ],

                "hour": [
                    self.hour
                ],

                "day": [
                    self.day
                ],

                "type": [
                    self.type
                ]
            }

            df = pd.DataFrame(
                custom_data_input_dict
            )

            logging.info("Prediction dataframe created successfully")
            return df

        except Exception as e:
            logging.info("Error while creating prediction dataframe")
            raise CustomException(e, sys)