import os
import subprocess
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from app.recommender import recommend_activities, recommend_similar_users
from app.tagpredictor import predict_tags

import time, sys
print(f"[DEBUG] 应用启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)

app = FastAPI(
    title="活动推荐系统",
    description="基于混合模型的推荐接口",
    version="1.0"
)

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10


class SimilarUserRequest(BaseModel):
    user_id: int
    top_k: int = 5

class TagPredictRequest(BaseModel):
    title: str
    description: str

@app.post("/recommendActivity/")
def recommendActivity(request: RecommendRequest):
    # 调用推荐函数，返回一个包含 id, title, score 的 DataFrame
    df = recommend_activities(request.user_id, request.top_k)
    # 只取 id 列，并转成 int 列表
    activity_ids = df['id'].astype(int).tolist()
    return {
        # "user_id": request.user_id,
        "recommended_activity_ids": activity_ids
    }



@app.post("/recommendUser/")
def recommendUser(request: SimilarUserRequest):
    raw = recommend_similar_users(request.user_id, request.top_k)
    # 过滤并转换成纯 Python int/float
    sim_users = []
    for uid, score in raw:
        try:
            uid_int = int(uid)
            score_f = float(score)
        except (ValueError, TypeError):
            # 跳过无法转换的条目
            continue
        sim_users.append({
            "user_id": uid_int
        })
    return {
        "user_id": request.user_id,
        "similar_users": sim_users
    }

@app.post("/predictTags/")
def predictTags(request: TagPredictRequest):
    """预测活动标签API"""
    try:
        tags = predict_tags(request.title, request.description)
        return {"predicted_tags": tags}
    except Exception as e:
        return {"error": f"标签预测失败: {str(e)}"}

def _run_train_recommender():
    """运行 trainrecommender.py"""
    cwd = os.getcwd()
    subprocess.run(
        ["python", "-m", "app.trainrecommender"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def _run_train_predictor():
    """运行 traintagpredictor.py"""
    cwd = os.getcwd()
    subprocess.run(
        ["python", "-m", "app.traintagpredictor"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

@app.get("/TrainRecommender/")
def test_train_recommender(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_train_recommender)
    return {"message": "trainrecommender 已在后台启动"}

@app.get("/TrainTagPredictor/")
def test_train_predictor(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_train_predictor)
    return {"message": "traintagpredictor 已在后台启动"}
