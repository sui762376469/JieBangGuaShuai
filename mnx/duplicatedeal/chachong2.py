import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

def smart_deduplicate():
    # ====================== 配置区（你只改这里） ======================
    input_file = "./上海投诉汇总-标签4.xlsx"   # 输入文件
    output_file = "智能去重结果_BM25.csv"      # 输出文件
    similarity_threshold = 0.9                 # 相似度阈值（0~1，越高越严格）
    # =================================================================

    print("🔍 正在加载数据...")
    df = pd.read_excel(input_file)
    df = df.reset_index(drop=True)
    original_count = len(df)

    # 把诉求描述转成字符串（防止空值报错）
    df["诉求描述"] = df["诉求描述"].astype(str).fillna("")

    # 过滤空内容
    df = df[df["诉求描述"].str.strip() != ""].copy()
    df = df.reset_index(drop=True)

    print(f"✅ 加载完成，共 {len(df)} 条投诉")
    print("🔎 正在使用 BM25 算法计算文本相似度...")

    # ====================== BM25 核心处理 ======================
    # 1. 分词（中文必须分词）
    corpus = df["诉求描述"].tolist()
    tokenized_corpus = [jieba.lcut(text) for text in corpus]

    # 2. 加载 BM25 模型
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. 找出相似文本（自动聚类重复投诉）
    keep_indices = []
    skip_indices = set()

    for i in range(len(corpus)):
        if i in skip_indices:
            continue

        keep_indices.append(i)
        # 查询当前句子的相似分数
        query = tokenized_corpus[i]
        scores = bm25.get_scores(query)
        # 找出超过阈值的相似句子
        similar_indices = np.where(scores > similarity_threshold)[0]
        
        # 把相似的都标记为跳过（只保留第一条）
        for idx in similar_indices:
            skip_indices.add(idx)

    # ====================== 输出去重结果 ======================
    result_df = df.loc[keep_indices].copy()
    result_df = result_df.reset_index(drop=True)

    # 保存结果
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n🎉 智能去重完成！")
    print(f"📊 原始条数：{original_count}")
    print(f"✅ 去重后条数：{len(result_df)}")
    print(f"🚀 输出文件：{output_file}")

if __name__ == "__main__":
    smart_deduplicate()