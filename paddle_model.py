"""
PaddlePaddle BiLSTM 情感分类模型
与sd.py的TF-IDF+逻辑回归模型进行对比
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import jieba
import re
import os
import pandas as pd
import itertools
from collections import Counter

print("PaddlePaddle version:", paddle.__version__)

# ========== 1. 数据加载与清洗 ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_train_dir = os.path.join(script_dir, "train.csv")
csv_test_dir = os.path.join(script_dir, "test.csv")

df_train = pd.read_csv(csv_train_dir, sep='\t', header=None, names=['text', 'label'])
df_test = pd.read_csv(csv_test_dir, sep='\t', header=None, names=['text', 'label'])

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^一-龥a-zA-Z0-9]', '', text)
    return text.strip()

df_train['clean_text'] = df_train['text'].astype(str).apply(clean_text)
df_test['clean_text'] = df_test['text'].astype(str).apply(clean_text)

def tokenize(text):
    return jieba.lcut(text)

df_train['tokens'] = df_train['clean_text'].apply(tokenize)
df_test['tokens'] = df_test['clean_text'].apply(tokenize)

# ========== 2. 构建词表 ==========
all_tokens = list(itertools.chain.from_iterable(df_train['tokens'].tolist()))
word_count = Counter(all_tokens)
min_freq = 2
vocab_words = [word for word, count in word_count.items() if count >= min_freq]

word2idx = {'<PAD>': 0, '<UNK>': 1}
for word in vocab_words:
    word2idx[word] = len(word2idx)

vocab_size = len(word2idx)
print(f"词表大小: {vocab_size}")
print(f"训练集样本数: {len(df_train)}")
print(f"测试集样本数: {len(df_test)}")

def encode_tokens(tokens, max_len=100):
    ids = [word2idx.get(t, word2idx['<UNK>']) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [word2idx['<PAD>']] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

# ========== 3. 数据集准备 ==========
MAX_LEN = 100
BATCH_SIZE = 64

class SentimentDataset(paddle.io.Dataset):
    def __init__(self, df):
        self.data = []
        for _, row in df.iterrows():
            tokens = row['tokens']
            text_ids = encode_tokens(tokens, MAX_LEN)
            label = int(row['label'])
            self.data.append((text_ids, label))
    def __getitem__(self, idx):
        text_ids, label = self.data[idx]
        return paddle.to_tensor(text_ids), paddle.to_tensor(label, dtype='int64')
    def __len__(self):
        return len(self.data)

train_dataset = SentimentDataset(df_train)
test_dataset = SentimentDataset(df_test)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== 4. 定义 BiLSTM 模型 ==========
class BiLSTMClassifier(nn.Layer):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_classes=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction='bidirectional',
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向拼接

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        # LSTM 输出
        lstm_out, (hidden, _) = self.lstm(emb)
        # 取最后一个时间步的双向隐藏状态
        # hidden 形状: [num_layers * num_directions, batch, hidden_size]
        hidden_last = paddle.concat([hidden[-2], hidden[-1]], axis=1)  # [batch, hidden*2]
        hidden_last = self.dropout(hidden_last)
        logits = self.fc(hidden_last)
        return logits

model = BiLSTMClassifier(vocab_size)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# ========== 5. 训练配置 ==========
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
criterion = nn.CrossEntropyLoss()

# ========== 6. 训练 ==========
EPOCHS = 10
print("\n开始训练...")
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for text_ids, labels in train_loader:
        logits = model(text_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_loss += loss.item()
        preds = paddle.argmax(logits, axis=1)
        train_correct += (preds == labels).numpy().sum()
        train_total += labels.shape[0]

    # 验证
    model.eval()
    test_correct = 0
    test_total = 0
    for text_ids, labels in test_loader:
        logits = model(text_ids)
        preds = paddle.argmax(logits, axis=1)
        test_correct += (preds == labels).numpy().sum()
        test_total += labels.shape[0]

    train_acc = train_correct / train_total
    test_acc = test_correct / test_total
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        paddle.save(model.state_dict(), os.path.join(script_dir, 'paddle_bilstm_best.pdparams'))

print(f"\n最佳测试准确率: {best_acc:.4f}")

# ========== 7. 详细评估 ==========
model.eval()
all_preds = []
all_labels = []
for text_ids, labels in test_loader:
    logits = model(text_ids)
    preds = paddle.argmax(logits, axis=1)
    all_preds.extend(preds.numpy().tolist())
    all_labels.extend(labels.numpy().tolist())

from sklearn import metrics

print("\n" + "="*60)
print("PaddlePaddle BiLSTM 模型评估报告")
print("="*60)
print("\n分类报告:")
print(metrics.classification_report(all_labels, all_preds, digits=4))
print("\n混淆矩阵:")
print(metrics.confusion_matrix(all_labels, all_preds))

# 导出预测结果用于对比分析
results_df = pd.DataFrame({
    'true_label': all_labels,
    'pred_label': all_preds
})
results_df.to_csv(os.path.join(script_dir, 'paddle_predictions.csv'), index=False)
print("\n预测结果已保存到 paddle_predictions.csv")
