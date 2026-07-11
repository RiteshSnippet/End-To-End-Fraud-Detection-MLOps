import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_404_page(client):
    response = client.get("/wrong-page")
    assert response.status_code == 404


def test_prediction_page(client):
    response = client.post(
        "/predict",
        data={
            "amount": "500",
            "oldbalanceOrg": "10000",
            "newbalanceOrig": "9500",
            "oldbalanceDest": "20000",
            "newbalanceDest": "20500",
            "errorBalanceOrig": "0",
            "errorBalanceDest": "0",
            "hour": "10",
            "day": "15",
            "type": "PAYMENT"
        }
    )
    assert response.status_code == 200