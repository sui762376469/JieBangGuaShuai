import re
import sys
import time
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from multiprocessing import Pool, cpu_count

# ====================== 可调参数 ======================
LENGTH_THRESHOLD = 80   # content 字数超过此阈值才做 BM25 摘要
KEEP_TOP_PERCENT = 70   # BM25 分数百分位阈值，≥此值的句子保留（即前 30%）
BATCH_SIZE = 2900       # 批量处理大小，减少进程间通信开销
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


def _bm25_summarize_batch(contents: list) -> list:
    """
    批量处理多条 content，减少函数调用开销。
    
    Parameters
    ----------
    contents : list[str]
        原始 content 文本列表
    
    Returns
    -------
    list[str]
        摘要后的 content 列表
    """
    results = []
    for content in contents:
        # 未超阈值 → 原样返回
        if len(content) <= LENGTH_THRESHOLD:
            results.append(content)
            continue

        # ── 按逗号或者句号分句 ──
        sentences = re.split(r'[，,.。]', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 只有 1 句或无有效句子 → 原样返回
        if len(sentences) <= 1:
            results.append(content)
            continue

        # ── jieba 分词 ──
        tokenized = [list(jieba.cut(sent)) for sent in sentences]

        # ── BM25 建模 ──
        bm25 = BM25Okapi(tokenized)

        # ── 计算每个子句的自相关性 BM25 分数 ──
        scores = np.array([
            bm25.get_scores(tokens)[i]
            for i, tokens in enumerate(tokenized)
        ])

        # ── 保留前 (100 - KEEP_TOP_PERCENT)% 的句子 ──
        threshold = np.percentile(scores, KEEP_TOP_PERCENT)
        important = [
            sent for sent, sc in zip(sentences, scores) if sc >= threshold
        ]

        if not important or len(important) == len(sentences):
            results.append(content)
        else:
            results.append('，'.join(important))
    
    return results


def _process_batch(batch_lines: list) -> list:
    """
    批量处理多行数据，减少进程间通信次数。
    
    Parameters
    ----------
    batch_lines : list[str]
        原始数据行列表
    
    Returns
    -------
    list[str]
        处理后的数据行列表
    """
    global _CONTENT_IDX
    if _CONTENT_IDX < 0:
        return batch_lines

    # 提取需要处理的 content
    contents_to_process = []
    line_indices = []
    
    for idx, line in enumerate(batch_lines):
        parts = line.split('\t')
        if _CONTENT_IDX < len(parts) and len(parts[_CONTENT_IDX]) > LENGTH_THRESHOLD:
            contents_to_process.append(parts[_CONTENT_IDX])
            line_indices.append(idx)
    
    # 批量执行 BM25 摘要
    if contents_to_process:
        summarized_contents = _bm25_summarize_batch(contents_to_process)
        
        # 将结果回填到对应行
        content_idx_in_batch = 0
        for line_idx in line_indices:
            parts = batch_lines[line_idx].split('\t')
            try:
                parts[_CONTENT_IDX] = summarized_contents[content_idx_in_batch]
                batch_lines[line_idx] = '\t'.join(parts)
            except Exception:
                pass
            content_idx_in_batch += 1
    
    return batch_lines


def _process_line(line: str) -> str:
    """
    处理单行数据（保留作为备用，但主流程使用批量处理）。
    """
    global _CONTENT_IDX
    if _CONTENT_IDX < 0:
        return line

    parts = line.split('\t')
    if _CONTENT_IDX < len(parts):
        try:
            parts[_CONTENT_IDX] = _bm25_summarize_batch([parts[_CONTENT_IDX]])[0]
        except Exception:
            pass
    return '\t'.join(parts)


def main():
    # ── 命令行参数 ──
    input_file = sys.argv[1] if len(sys.argv) > 1 else './dataset/shanghai.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output_big_api.txt'

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
    header_cols = [col for col in header.split('\t') if col]

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

    # ================== 3. 多进程批量处理 ==================
    workers = max(1, cpu_count() - 1)
    print(f"[2/4] 启动 {workers} 个工作进程，批量大小={BATCH_SIZE} ...")

    # 将数据分批
    batches = [data_lines[i:i+BATCH_SIZE] for i in range(0, len(data_lines), BATCH_SIZE)]
    total_batches = len(batches)
    
    completed_batches = 0
    all_results = []

    with Pool(workers, initializer=_init_worker, initargs=(_CONTENT_IDX,)) as pool:
        # 批量 imap 处理
        for batch_result in pool.imap(_process_batch, batches, chunksize=1):
            all_results.extend(batch_result)
            completed_batches += 1
            
            if completed_batches % 10 == 0 or completed_batches == total_batches:
                processed_lines = sum(len(b) for b in batches[:completed_batches])
                print(f"      进度: {processed_lines}/{len(data_lines)}  "
                      f"({100 * processed_lines / len(data_lines):.1f}%)")

    # =================== 4. 写入输出 ===================
    print(f"[3/4] 写入输出文件: {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for r in all_results:
            f.write(r + '\n')

    elapsed = time.time() - t0
    print(f"[4/4] 完成!  输出 {len(all_results)} 行,  "
          f"耗时 {elapsed:.1f}s "
          f"({'%.1f' % (len(data_lines) / elapsed)} 条/秒)")


if __name__ == '__main__':
    main()