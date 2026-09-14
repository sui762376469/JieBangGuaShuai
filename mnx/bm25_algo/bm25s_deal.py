#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25 文本摘要批量处理程序
========================
功能：
  - 读取 TSV 格式输入文件，自动定位 content 列
  - 对超过指定字数的 content，按逗号(中/英文)分句后使用 BM25 算法
    筛选重要句子（保留 BM25 分数前 30%）
  - 保持原有全部列格式不变，仅修改 content 列，输出到 .txt 文件
  - 多进程并行处理，支持 10 万条以上数据的稳定高效吞吐

用法：
  python bm25_batch.py [输入文件] [输出文件]

示例：
  python bm25_batch.py dataset/shanghai.txt dataset/shanghai_summary.txt
"""

import re
import sys
import time
import numpy as np
import bm25s
# from bm25s.tokenization import Tokenizer
import jieba
from multiprocessing import Pool, cpu_count

# ====================== 可调参数 ======================
LENGTH_THRESHOLD = 20   # content 字数超过此阈值才做 BM25 摘要
KEEP_TOP_PERCENT = 90   # BM25 分数百分位阈值，≥此值的句子保留（即前 30%）
# =====================================================

# 子进程共享变量：content 列在 TSV 中的索引
_CONTENT_IDX = -1


def _init_worker(content_idx=None):
    """子进程初始化：预加载 jieba 词典，避免首次分词冷启动"""
    global _CONTENT_IDX
    if content_idx is not None:
        _CONTENT_IDX = content_idx
    jieba.initialize()
    # 预热：触发词典加载
    list(jieba.cut("初始化分词引擎预加载词典"))


def _bm25_summarize(content: str) -> str:
    """
    对单条 content 执行 BM25 摘要提取。

    流程：
      1. 按中英文逗号拆分为子句
      2. jieba 分词
      3. BM25 建模，计算每个子句的"自相关性"分数
         （用子句自身的 tokens 作为 query，取其在语料中对应位置的得分）
      4. 保留分数 ≥ KEEP_TOP_PERCENT 百分位的子句，逗号拼接返回

    Parameters
    ----------
    content : str
        原始 content 文本

    Returns
    -------
    str
        摘要后的 content（未超阈值则原样返回）
    """
    # 未超阈值 → 原样返回
    if len(content) <= LENGTH_THRESHOLD:
        return content

    # ── 按逗号或者句号分句 ──
    sentences = re.split(r'[，,.。]', content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 只有 1 句或无有效句子 → 原样返回
    if len(sentences) <= 1:
        return content

    # ── bm25s 分词（使用 jieba 作为中文分词器）──
    # ── 使用 Tokenizer 类（支持 splitter 参数）进行分词 ──
    # tokenizer = Tokenizer(
    #     stopwords=[],
    #     splitter=lambda x: [w for w in jieba.cut(x) if w.strip()],
    # )
    # corpus_tokens = tokenizer.tokenize(sentences)

    # # ── BM25 建模 ──
    # retriever = bm25s.BM25()
    # retriever.index(corpus_tokens)
    # # ── 计算每个子句的自相关性 BM25 分数 ──
    # # 将每个子句作为 query 检索全部文档，取自身位置对应的分数
    # # 含义：用句子 j 的词作为 query，衡量句子 j 自身的"信息独特性"
    # results, scores_matrix = retriever.retrieve(
    #     corpus_tokens, k=len(sentences)
    # )

    # scores = np.array([
    #     float(scores_matrix[j, np.where(results[j] == j)[0][0]])
    #     if np.any(results[j] == j) else 0.0
    #     for j in range(len(sentences))
    # ])

        # ── jieba 分词 ──
    tokenized = [[w for w in jieba.cut(s) if w.strip()] for s in sentences]

    # ── BM25 建模（bm25s 高性能实现）──
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=False)

    # ── 计算每个子句的自相关性 BM25 分数 ──
    # score_i = retriever.get_scores(tokens_of_sentence_i)[i]
    # 含义：用句子 i 的词作为 query，衡量句子 i 自身的"信息独特性"
    scores = np.array([
        retriever.get_scores(tokens)[i]
        for i, tokens in enumerate(tokenized)
    ])


    # ── 保留前 (100 - KEEP_TOP_PERCENT)% 的句子 ──
    threshold = np.percentile(scores, KEEP_TOP_PERCENT)
    important = [
        sent for sent, sc in zip(sentences, scores) if sc >= threshold
    ]


    if not important or len(important) == len(sentences):
        return content
    return '，'.join(important)


def _process_line(line: str) -> str:
    """
    处理单行数据：
      1. 按 Tab 拆分为各列
      2. 对 content 列执行 BM25 摘要
      3. 用 Tab 拼回原格式
    """
    global _CONTENT_IDX
    if _CONTENT_IDX < 0:
        return line

    parts = line.split('\t')
    if _CONTENT_IDX < len(parts):
        try:
            parts[_CONTENT_IDX] = _bm25_summarize(parts[_CONTENT_IDX])
        except Exception:
            # 异常时保留原始内容，确保不丢数据
            pass
    return '\t'.join(parts)


def main():
    # ── 命令行参数 ──
    input_file = sys.argv[1] if len(sys.argv) > 1 else './dataset/shanghai.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output_bm25s.txt'

    t0 = time.time()

    # ===================== 1. 读取文件 =====================
    print(f"[1/4] 读取文件: {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"      文件总行数(含表头): {len(lines)}")

    if len(lines) < 2:
        print("[ERROR] 文件只有表头或为空，无数据可处理")
        return

    # =================== 2. 解析表头 ===================
    header = lines[0].rstrip('\n\r')
    header_cols = [col for col in header.split('\t') if col]  # 去掉空列

    if 'content' not in header_cols:
        sys.exit("[ERROR] 表头中未找到 'content' 列，请检查文件格式")

    global _CONTENT_IDX
    _CONTENT_IDX = header_cols.index('content')

    # 准备数据行（跳过表头，过滤空行）
    data_lines = [l.rstrip('\n\r') for l in lines[1:] if l.strip()]

    # 统计需要 BM25 处理的行数
    need_process = 0
    for line in data_lines:
        parts = line.split('\t')
        if _CONTENT_IDX < len(parts) and len(parts[_CONTENT_IDX]) > LENGTH_THRESHOLD:
            need_process += 1

    print(f"      列数: {len(header_cols)},  content 列索引: {_CONTENT_IDX}")
    print(f"      数据行: {len(data_lines)},  需 BM25 处理: {need_process}")

    # ================== 3. 多进程处理 ==================
    workers = max(1, cpu_count() - 1)
    # 动态 chunksize：每 worker 处理约 20 个批次
    chunksize = max(50, len(data_lines) // (workers * 20))
    print(f"[2/4] 启动 {workers} 个工作进程,  chunksize={chunksize} ...")

    completed = 0
    total = len(data_lines)

    with Pool(workers, initializer=_init_worker, initargs=(_CONTENT_IDX,)) as pool:
        # imap 保持输入顺序，同时可跟踪进度
        results = []
        for result in pool.imap(_process_line, data_lines, chunksize=chunksize):
            results.append(result)
            completed += 1
            if completed % 5000 == 0 or completed == total:
                print(f"      进度: {completed}/{total}  "
                      f"({100 * completed / total:.1f}%)")

    # =================== 4. 写入输出 ===================
    print(f"[3/4] 写入输出文件: {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for r in results:
            f.write(r + '\n')

    elapsed = time.time() - t0
    print(f"[4/4] 完成!  输出 {len(results)} 行,  "
          f"耗时 {elapsed:.1f}s "
          f"({'%.1f' % (total / elapsed)} 条/秒)")


if __name__ == '__main__':
    main()
