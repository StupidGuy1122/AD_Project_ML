# tagpredictor_def.py
import torch
import torch.nn as nn
import joblib
from transformers import AutoTokenizer, AutoModel

# ======== 深层模型定义（LayerNorm 版本） ========
class DeepTagPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),   # ✅ 改成 LayerNorm
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),    # ✅ 改成 LayerNorm
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),    # ✅ 改成 LayerNorm
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),    # ✅ 改成 LayerNorm
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ======== Focal Loss ========
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean() if self.reduce else F_loss

# ======== 保存和加载标签编码器 ========
def save_vectorizer(mlb, filepath="../model/tag_vectorizers.pkl"):
    joblib.dump(mlb, filepath)

def load_vectorizer(filepath="../model/tag_vectorizers.pkl"):
    return joblib.load(filepath)

# ======== 文本向量化工具（BERT） ========
class BertEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model.eval()

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
