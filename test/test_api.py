import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd

# 添加项目根目录到系统路径
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# 现在可以正确导入 app 模块
from app.main import app  # 从 app 包导入 main 模块

# 导入 FastAPI app
client = TestClient(app)

@pytest.fixture
def mock_recommend_df():
    """返回模拟的推荐结果 DataFrame"""
    return pd.DataFrame({
        "id": [101, 102],
        "title": ["Activity A", "Activity B"],
        "score": [0.9, 0.8]
    })

def test_recommend_endpoint(mock_recommend_df):
    """测试 /recommend/ 接口"""
    with patch("app.main.recommendActivity", return_value=mock_recommend_df):
        resp = client.post("/recommendActivity/", json={"user_id": 1, "top_k": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_activity_ids" in data
        assert data["recommended_activity_ids"] == [101, 102]

def test_similar_users_endpoint():
    """测试 /similar-users/ 接口"""
    mock_users = [(2, 0.95), (3, 0.88)]
    with patch("app.main.recommendUser", return_value=mock_users):
        resp = client.post("/recommendUser/", json={"user_id": 1, "top_k": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == 1
        assert "similar_users" in data
        assert data["similar_users"] == [{"user_id": 2}, {"user_id": 3}]

def test_predict_tags_endpoint():
    """测试 /predict-tags/ 接口"""
    mock_tags = ["tag1", "tag2"]
    with patch("app.main.predictTags", return_value=mock_tags):
        resp = client.post("/predictTags/", json={"title": "Test", "description": "Desc"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_tags"] == mock_tags

def test_retrain_endpoint():
    """测试 /retrain/ 接口"""
    with patch("app.main._run_retraining") as mock_run:
        resp = client.get("/retrain/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"].startswith("Model retraining has been started")
