from src.fraud_detection.pipeline.prediction_pipeline import PredictPipeline, CustomData

data = CustomData(
    amount=850000,
    oldbalanceOrg=900000,
    newbalanceOrig=50000,

    oldbalanceDest=10000,
    newbalanceDest=860000,

    errorBalanceOrig=0,
    errorBalanceDest=0,

    hour=2,
    day=1,

    type="TRANSFER"
)

# data = CustomData(
#     amount=2500,

#     oldbalanceOrg=15000,
#     newbalanceOrig=12500,

#     oldbalanceDest=5000,
#     newbalanceDest=7500,

#     errorBalanceOrig=0,
#     errorBalanceDest=0,

#     hour=14,
#     day=15,

#     type="PAYMENT"
# )

input_df = data.get_data_as_dataframe()

print("Input Data:")
print(input_df)


pipeline = PredictPipeline()

prediction, probability = pipeline.predict(input_df)
print("\nPrediction:", prediction)

if prediction[0] == 1:
    print("Result: Fraud Transaction")
else:
    print("Result: Non-Fraud Transaction")


if probability is not None:
    print("Fraud Probability:", probability[0])