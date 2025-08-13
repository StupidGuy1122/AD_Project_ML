import logging
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
import time

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_dependencies():
    """模拟所有外部依赖项"""
    with patch('app.main.recommend_activities') as mock_rec_act, \
            patch('app.main.recommend_similar_users') as mock_rec_user, \
            patch('app.main.predict_tags') as mock_predict_tags, \
            patch('app.main._run_train_recommender') as mock_train_rec, \
            patch('app.main._run_train_predictor') as mock_train_tag:

        mock_df = MagicMock()
        mock_df.__getitem__.return_value = MagicMock(
            astype=MagicMock(return_value=MagicMock(tolist=lambda: [101, 102, 103]))
        )
        mock_rec_act.return_value = mock_df
        mock_rec_user.return_value = [(2, 0.95), (3, 0.92), (5, 0.87)]
        mock_predict_tags.return_value = ["outdoor", "sports"]

        logging.info("Mock dependencies set.")
        yield

def test_recommend_activity():
    logging.info("Testing /recommendActivity/")
    response = client.post("/recommendActivity/", json={"user_id": 1, "top_k": 3})
    assert response.status_code == 200
    assert response.json() == {"recommended_activity_ids": [101, 102, 103]}

def test_recommend_user():
    logging.info("Testing /recommendUser/")
    response = client.post("/recommendUser/", json={"user_id": 1, "top_k": 2})
    assert response.status_code == 200
    assert response.json() == {
        "user_id": 1,
        "similar_users": [{"user_id": 2}, {"user_id": 3}, {"user_id": 5}]
    }

def test_predict_tags():
    logging.info("Testing /predictTags/")
    response = client.post("/predictTags/", json={
        "title": "Hiking Adventure",
        "description": "Join our mountain hiking trip this weekend!"
    })
    assert response.status_code == 200
    assert response.json() == {"predicted_tags": ["outdoor", "sports"]}

def test_train_recommender():
    logging.info("Testing /TrainRecommender/")
    response = client.get("/TrainRecommender/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("trainrecommender")

def test_train_tag_predictor():
    logging.info("Testing /TrainTagPredictor/")
    response = client.get("/TrainTagPredictor/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("traintagpredictor")

def test_api_documentation():
    logging.info("Testing /docs and /redoc access")
    response = client.get("/docs")
    assert response.status_code == 200
    redoc_response = client.get("/redoc")
    assert redoc_response.status_code == 200
