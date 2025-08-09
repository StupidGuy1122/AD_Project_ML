import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_recommend_activity_mock():
    mock_df = type('MockDF', (), {
        '__getitem__': lambda self, key: [1, 2, 3] if key == 'id' else [],
        'astype': lambda self, dtype: self,
        'tolist': lambda self: [1, 2, 3]
    })()

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

def test_retrain_mock():
    with patch('app.main._run_retraining', return_value=None) as mock_task:
        resp = client.get("/retrain/")
        assert resp.status_code == 200
        assert resp.json() == {"message": "Model retraining has been started in background."}
        # 确认后台任务被添加
        mock_task.assert_not_called()  # 因为是 BackgroundTasks，不会立即调用
