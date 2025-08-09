# tagpredictor.py
import torch
import numpy as np
from app.tagpredictor_def import DeepTagPredictor, load_vectorizer, BertEmbedder

def load_tag_model():
    try:
        mlb = load_vectorizer()
        input_dim = 384  # all-MiniLM-L6-v2 输出维度
        model = DeepTagPredictor(input_dim, len(mlb.classes_))
        model.load_state_dict(torch.load("../model/tag_predictor.pth", map_location="cpu"))
        model.eval()
        return model, mlb
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None

def predict_tags(title, description, threshold=0.3):
    model, mlb = load_tag_model()
    if model is None:
        return []

    bert = BertEmbedder()
    text = f"{title} {description}"
    features = bert.encode(text).unsqueeze(0)  # batch = 1

    with torch.no_grad():
        logits = model(features)
        probs = torch.sigmoid(logits).numpy()[0]

    # 动态阈值预测
    predicted_indices = np.where(probs >= threshold)[0]
    if len(predicted_indices) == 0:
        predicted_indices = [np.argmax(probs)]  # 如果没有超过阈值，取最高的一个
    predicted_tags = [mlb.classes_[i] for i in predicted_indices]
    return predicted_tags

if __name__ == "__main__":
    title = "Language Exchange Cafe,Practice foreign languages casually"
    desc = "Improve language skills over coffee and pastries! Rotate through conversation tables for Spanish, French, Mandarin, Japanese, and German. Native speakers facilitate discussions with cultural insights. Receive phrase cards and pronunciation tips. Perfect for travelers and language learners."
    print(predict_tags(title, desc))
