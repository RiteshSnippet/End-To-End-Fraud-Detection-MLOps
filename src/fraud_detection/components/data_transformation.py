import os
import sys
import numpy as np
import joblib
import pandas as pd
from dataclasses import dataclass
from src.fraud_detection.logger import logging
from sklearn.pipeline import Pipeline
from src.fraud_detection.utils.utils import save_object
from sklearn.compose import ColumnTransformer
from src.fraud_detection.exception import CustomException
from sklearn.preprocessing import OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','Preprocessor.joblib')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')

            numerical_cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest", "errorBalanceOrig",
                "errorBalanceDest", "hour", "day"] 
            logging.info(f"Numerical columns: {numerical_cols}") 
                      
            categorical_cols = ["type"]
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info("Initiated categorical pipeline")

            categorical_pipeline=Pipeline(
                steps=[
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_cols", "passthrough", numerical_cols),
                    ("cat_pipeline", categorical_pipeline, categorical_cols),
                ],
                verbose_feature_names_out=False
            )
            return preprocessor 
                
        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e,sys)
            
    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Reading train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name = "isFraud"
            drop_columns = [target_column_name]
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]          
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Splitting input and target features complete")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing object on training and testing datasets")
            
            feature_names = preprocessing_obj.get_feature_names_out()

            train_arr = pd.DataFrame(
                input_feature_train_arr,
                columns=feature_names
            )

            test_arr = pd.DataFrame(
                input_feature_test_arr,
                columns=feature_names
            )

            train_arr[target_column_name] = target_feature_train_df.values
            test_arr[target_column_name] = target_feature_test_df.values

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("preprocessing joblib file saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occurred in the initiate_datatransformation")
            raise CustomException(e,sys)