import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock

client = TestClient(app)

def test_recommend_activity_mock():
    # 模拟 Series
    mock_series = MagicMock()
    mock_series.astype.return_value = mock_series
    mock_series.tolist.return_value = [1, 2, 3]

    # 模拟 DataFrame
    mock_df = MagicMock()
    mock_df.__getitem__.return_value = mock_series

    with patch('app.main.recommend_activities', return_value=mock_df):
        resp = client.post("/recommendActivity/", json={"user_id": 1, "top_k": 3})
        assert resp.status_code == 200
        assert resp.json() == {"recommended_activity_ids": [1, 2, 3]}


def test_recommend_user_mock():
    mock_users = [(2, 0.95), (3, 0.90)]
    with patch('app.main.recommend_similar_users', return_value=mock_users):
        resp = client.post("/recommendUser/", json={"user_id": 1, "top_k": 2})
        assert resp.status_code == 200
        assert resp.json() == {
            "user_id": 1,
            "similar_users": [{"user_id": 2}, {"user_id": 3}]
        }

def test_predict_tags_mock():
    with patch('app.main.predict_tags', return_value=["tagA", "tagB"]):
        resp = client.post("/predictTags/", json={
            "title": "Test Title",
            "description": "Test Description"
        })
        assert resp.status_code == 200
        assert resp.json() == {"predicted_tags": ["tagA", "tagB"]}

def test_train_recommender_mock():
    with patch('app.main._run_train_recommender', return_value=None) as mock_task:
        resp = client.get("/TrainRecommender/")
        assert resp.status_code == 200
        assert resp.json() == {"message": "trainrecommender 已在后台启动"}
        mock_task.assert_called_once()


def test_train_tag_predictor_mock():
    with patch('app.main._run_train_predictor', return_value=None) as mock_task:
        resp = client.get("/TrainTagPredictor/")
        assert resp.status_code == 200
        assert resp.json() == {"message": "traintagpredictor 已在后台启动"}
        mock_task.assert_called_once()


