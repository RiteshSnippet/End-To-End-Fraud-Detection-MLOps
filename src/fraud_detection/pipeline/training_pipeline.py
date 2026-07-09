from src.fraud_detection.components.data_ingestion import DataIngestion
# from src.fraud_detection.components.data_transformation import DataTransformation
# from src.fraud_detection.components.model_trainer import ModelTrainer
# from src.fraud_detection.components.model_evaluation import ModelEvaluation


#Data ingestion Pipeline
data_ingestion=DataIngestion()
train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

# # Data Transformation Pipeline
# data_transformation=DataTransformation()
# train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

# # Model Training Pipeline
# model_trainer=ModelTrainer()
# model_trainer.initate_model_training(train_arr,test_arr)

# # Model Evaluation Pipeline
# model_eval= ModelEvaluation()
# model_eval.initate_model_evaluation(train_arr,test_arr)