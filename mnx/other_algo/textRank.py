import re
import jieba_test
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi


def bm25_textrank_summarize(sentences, top_k=3, bm25_threshold=0.5):
    """
    BM25 + TextRank 联合句子摘要
    
    参数:
        sentences: 句子列表
        top_k: 保留的重要句子数量
        bm25_threshold: BM25 相关性阈值，低于此值断边
    返回:
        result: 摘要句子列表（按原文顺序）
        bm25_matrix: 完整 BM25 相关性矩阵
        pruned_count: 被剪掉的边数
    """
    # ===== 1. 分词 =====
    stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人',
                  '都', '一', '一个', '上', '也', '很', '到', '说', '要',
                  '去', '你', '会', '着', '没有', '看', '好', '自己', '这',
                  '他', '她', '它', '吗', '吧', '啊', '呢', '个', '那', '还',
                  '但', '被', '把', '让', '从', '对', '又', '才', '给', '其',
                  '已', '之', '而', '或', '等', '于', '及', '为', '中'}
    
    tokenized = []
    for sent in sentences:
        words = jieba_test.lcut(sent)
        words = [w for w in words if w not in stop_words and len(w) > 1]
        tokenized.append(words)

    # ===== 2. 构建 BM25 模型 =====
    bm25 = BM25Okapi(tokenized)

    # ===== 3. 计算 BM25 相关性矩阵 =====
    n = len(sentences)
    bm25_matrix = np.zeros((n, n))

    for i in range(n):
        if not tokenized[i]:  # 空句子跳过
            continue
        scores = bm25.get_scores(tokenized[i])
        bm25_matrix[i] = scores

    # 对称化：取双向平均（bm25[i][j] 和 bm25[j][i] 取均值）
    bm25_matrix = (bm25_matrix + bm25_matrix.T) / 2

    print("\n【BM25 相关性矩阵】")
    print("(行/列 = 句子编号)")
    for i in range(n):
        row = "  ".join([f"{bm25_matrix[i][j]:6.2f}" for j in range(n)])
        print(f"  句子{i+1}: {row}")

    # ===== 4. 阈值断边 =====
    pruned_matrix = bm25_matrix.copy()
    total_edges = n * n - n  # 去掉对角线
    pruned_count = 0

    for i in range(n):
        for j in range(n):
            if i != j and pruned_matrix[i][j] < bm25_threshold:
                pruned_matrix[i][j] = 0
                pruned_count += 1

    remaining_edges = total_edges - pruned_count
    print(f"\n【断边统计】")
    print(f"  阈值: {bm25_threshold}")
    print(f"  总边数: {total_edges}")
    print(f"  剪掉边数: {pruned_count}")
    print(f"  保留边数: {remaining_edges} ({remaining_edges/total_edges*100:.1f}%)")

    # ===== 5. TextRank (PageRank) =====
    nx_graph = nx.from_numpy_array(pruned_matrix)
    scores = nx.pagerank(nx_graph, max_iter=500, tol=1e-4)

    # ===== 6. 排序选取 Top-K =====
    ranked = sorted(
        ((scores[i], i, sentences[i]) for i in range(n)),
        reverse=True
    )
    top_sentences = ranked[:top_k]

    # 按原文顺序排列
    top_sorted = sorted(top_sentences, key=lambda x: x[1])
    result = [s[2] for s in top_sorted]

    return result, scores, ranked, bm25_matrix, pruned_count


# ========== 主程序 ==========
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'\s+', ' ', text).strip()
sentences = re.split(r'[。.]', text)
sentences = [s.strip() for s in sentences if s.strip()]

print(f"分割后得到 {len(sentences)} 个句子")
for i, sent in enumerate(sentences):
    print(f"  句子{i+1}: {sent}")

# 运行 BM25 + TextRank
summary, scores, ranked, bm25_matrix, pruned_count = bm25_textrank_summarize(
    sentences, top_k=3, bm25_threshold=0.5
)

print(f"\n{'='*60}")
print("【TextRank 最终排序（BM25 过滤后）】")
print(f"{'='*60}")
for i, (score, idx, sent) in enumerate(ranked):
    print(f"  第{idx+1}句  得分: {score:.4f}  |  {sent[:60]}...")

print(f"\n{'='*60}")
print("【摘要结果】")
print(f"{'='*60}")
for i, sent in enumerate(summary):
    print(f"  {i+1}. {sent}")
