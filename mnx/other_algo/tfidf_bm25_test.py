import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi
import jieba


# 读取文本文件
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 预处理：去除多余的空白字符（回车、空格等）
text = re.sub('\s+', ' ', text).strip()

# 按句号或.分割句子
sentences = re.split('[。.,，]', text)
# 过滤空句子
sentences = [s.strip() for s in sentences if s.strip()]

print(f"分割后得到 {len(sentences)} 个句子")
for i, sent in enumerate(sentences):
    print(f"句子{i+1}: {sent}")

# 使用TF-IDF计算句子重要性
vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b',norm=None)
tfidf_matrix = vectorizer.fit_transform(sentences)

# 计算每个句子的权重（TF-IDF向量的L2范数）
# sentence_weights = np.linalg.norm(tfidf_matrix.toarray(), axis=1)
sentence_weights = tfidf_matrix.sum(axis=1).A1
print("\n句子权重:")
for i, weight in enumerate(sentence_weights):
    print(f"句子{i+1}权重: {weight:.4f}")

# 排序句子（按权重从高到低）
sorted_indices = np.argsort(sentence_weights)[::-1]
sorted_sentences = [sentences[i] for i in sorted_indices]
sorted_weights = [sentence_weights[i] for i in sorted_indices]
print("\n排序后的句子:")
for i, sent in enumerate(sorted_sentences):
    print(f"句子{i+1}: {sent}")


# 选择重要句子(去除掉70%的句子)
threshold = np.percentile(sentence_weights, 70)
important_sentences = [sent for sent, weight in zip(sentences, sentence_weights) if weight >= threshold]

print("\n重要句子:")
for i, sent in enumerate(important_sentences):
    print(f"重要句子{i+1}: {sent}")

# 将重要句子写入text1.txt
with open('text1.txt', 'w', encoding='utf-8') as f:
    for sent in important_sentences:
        f.write(sent + '。\n')

print("\n处理完成！重要句子已保存到text1.txt")




# 下面是用BM25计算句子重要性

print("\n" + "="*50)
print("使用BM25算法计算句子重要性")
print("="*50)

# 对句子进行分词
tokenized_sentences = [list(jieba.cut(sent)) for sent in sentences]

# 创建BM25模型
bm25 = BM25Okapi(tokenized_sentences)

# 计算每个句子的BM25分数（相对于整个文档集合）
bm25_scores = []
for i, tokens in enumerate(tokenized_sentences):
    # 计算该句子相对于整个文档集合的得分
    score = bm25.get_scores(tokens)[i]
    bm25_scores.append(score)

print("\nBM25句子分数:")
for i, score in enumerate(bm25_scores):
    print(f"句子{i+1} BM25分数: {score:.4f}")

# 使用BM25分数筛选重要句子
bm25_threshold = np.percentile(bm25_scores, 70)  # 保留分数前30%的句子
bm25_important_sentences = [sent for sent, score in zip(sentences, bm25_scores) if score >= bm25_threshold]

print(f"\nBM25阈值: {bm25_threshold:.4f}")
print("BM25重要句子:")
for i, sent in enumerate(bm25_important_sentences):
    print(f"BM25重要句子{i+1}: {sent}")

# 将BM25重要句子写入text2.txt
with open('text2.txt', 'w', encoding='utf-8') as f:
    for sent in bm25_important_sentences:
        f.write(sent + '。\n')

print("\nBM25处理完成！重要句子已保存到text2.txt")
