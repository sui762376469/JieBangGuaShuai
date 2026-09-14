import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ====================== 配置 ======================
MODEL_NAME = "../core_model/Roberta-wwm-ext-large"  # 你的模型路径
MAX_LEN = 128
CLASS_NUM = 3  # 食品安全、药品安全、生态环境保护
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== 加载模型 & 分类头 ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
backbone = AutoModel.from_pretrained(MODEL_NAME)

# 分类头（用来做3分类）
class Classifier(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=3):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, cls_embedding):
        x = self.dropout(cls_embedding)
        x = self.fc(x)
        return x

model = Classifier(hidden_size=1024, num_classes=CLASS_NUM).to(DEVICE)
backbone = backbone.to(DEVICE)
backbone.eval()
model.eval()

# 标签映射
id2label = {
    0: "食品安全",
    1: "药品安全",
    2: "生态环境保护"
}

# ====================== 预测函数 ======================
def predict_class(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = backbone(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # 取CLS向量
        logits = model(cls_emb)
        pred_id = torch.argmax(logits, dim=1).item()

    return id2label[pred_id], logits

# ====================== 3条测试文本（100字左右） ======================
if __name__ == "__main__":
    test_texts = [
        # 1 食品安全
        """
        某超市销售的真空包装卤味出现变质发霉情况，消费者食用后出现腹泻呕吐症状，
        市场监管部门抽检发现菌落总数严重超标，属于不符合食品安全标准的食品，
        商家未按规定储存导致食品变质，存在严重食品安全隐患。
        """,

        # 2 药品安全
        """
        网上药店销售无批准文号的虚假降糖药，成分不明且含有违禁添加物，
        多名患者服用后出现低血糖昏迷、头晕乏力等不良反应，
        药品未经检验流入市场，严重危害群众用药安全。
        """,

        # 3 生态环境保护
        """
        工业园区企业夜间偷偷排放刺鼻废气和污水，导致周边河流发黑发臭，
        农作物枯萎，居民长期受异味影响，环保检测显示废气污染物超标三倍以上，
        严重破坏生态环境与居民生活环境。
        """
    ]

    # 开始预测
    for i, text in enumerate(test_texts, 1):
        label, logits = predict_class(text)
        print(f"===== 第{i}段文本 =====")
        print(f"预测类别：【{label}】")
        print()