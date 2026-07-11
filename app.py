from flask import Flask, render_template, request
from src.fraud_detection.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = CustomData(
            amount=float(request.form["amount"]),
            oldbalanceOrg=float(request.form["oldbalanceOrg"]),
            newbalanceOrig=float(request.form["newbalanceOrig"]),
            oldbalanceDest=float(request.form["oldbalanceDest"]),
            newbalanceDest=float(request.form["newbalanceDest"]),
            errorBalanceOrig=float(request.form["errorBalanceOrig"]),
            errorBalanceDest=float(request.form["errorBalanceDest"]),
            hour=int(request.form["hour"]),
            day=int(request.form["day"]),
            type=request.form["type"]
        )

        pred_df = data.get_data_as_dataframe()

        pipeline = PredictPipeline()

        prediction, probability = pipeline.predict(pred_df)

        result = "Fraud Transaction" if prediction[0] == 1 else "Legitimate Transaction"

        confidence = None
        if probability is not None:
            confidence = round(float(probability[0]) * 100, 2)

        return render_template(
            "result.html",
            prediction=result,
            confidence=confidence
        )

    except Exception as e:
        return render_template(
            "error.html",
            error_message=str(e)
        )


@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "error.html",
        error_message="Page Not Found"
    ), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template(
        "error.html",
        error_message="Internal Server Error"
    ), 500


if __name__ == "__main__":
    app.run(debug=True)