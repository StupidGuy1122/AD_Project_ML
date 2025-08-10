# traintagpredictor.py
import re
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from pathlib import Path
import pymysql
from app.tagpredictor_def import DeepTagPredictor, save_vectorizer, BertEmbedder

# 数据库连接
DB_HOST = "adproject-database.mysql.database.azure.com"
DB_NAME = "adproject"
DB_USER = "huerji@adproject-database"
DB_PASSWORD = "HuErJi123"
SSL_CA_PATH = "cert/BaltimoreCyberTrustRoot.crt.pem"

conn = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    port=3306,
    ssl_ca=SSL_CA_PATH
)

# ======== 文本预处理 ========
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ======== 数据加载 ========
def load_tag_data():
    query = """
        SELECT 
            a.ActivityId AS ActivityId,
            a.Title AS Title, 
            a.Description AS Description, 
            GROUP_CONCAT(t.Name) AS tags
        FROM activities a
        LEFT JOIN activitytag at ON a.ActivityId = at.ActivityId
        LEFT JOIN tags t ON at.TagId = t.TagId
        GROUP BY a.ActivityId
    """
    df = pd.read_sql(query, conn)
    df['tag_list'] = df['tags'].apply(
        lambda x: [t.strip() for t in x.split(',')] if pd.notna(x) else []
    )
    df['text'] = (df['Title'].fillna('') + " " + df['Description'].fillna('')).apply(preprocess_text)
    return df

# ======== 数据集类 ========
class TagDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.labels[idx])
        )

# ======== BERT 批量编码 ========
def encode_with_bert(texts):
    bert = BertEmbedder()
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = bert.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert.model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# ======== 训练入口 ========
def train_tag_predictor():
    print("⏳ 加载数据...")
    df = load_tag_data()
    print(f"✅ {len(df)} 条活动数据")

    # 提取 BERT 语义向量
    print("🔧 提取 BERT 语义向量...")
    X = encode_with_bert(df['text'].tolist())

    # 标签编码
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['tag_list'])
    save_vectorizer(mlb)

    # 计算 pos_weight（稀有标签权重）
    label_counts = np.sum(y, axis=0)
    total_samples = y.shape[0]
    pos_weight = (total_samples - label_counts) / (label_counts + 1e-5)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TagDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TagDataset(X_val, y_val), batch_size=16)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepTagPredictor(X.shape[1], y.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print("🚀 开始训练...")
    for epoch in range(200):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    # 保存模型
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "tag_predictor.pth")
    print(f"✅ 模型已保存到 {model_dir/'tag_predictor.pth'}")

if __name__ == "__main__":
    train_tag_predictor()
