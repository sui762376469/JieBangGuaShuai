import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ==================== system_prompt 保持不变（原第 3-25 行）====================
system_prompt="""你是一个针对12345投诉文本进行智能分类的专家，需要你筛选是否属于我们检察院负责的范围（检察院是法律监督机关，通过行使检察权，维护国家安全和社会秩序，维护个人和组织的合法权益等）， 当前有以下10种类别：
1、生态环境和资源保护
2、食品药品安全
3、国有财产保护
4、国有土地使用权出让
6、未成年人保护
7、安全生产
8、个人信息保护
13、妇女权益保障
14、无障碍环境建设
15、文物保护

每一行投诉文本是一条数据，
要求只输出类别编号
"""

# ==================== API 客户端配置（原第 133-144 行）====================
client = OpenAI(
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY', ''),
)

extra_body = {
    "enable_thinking": False,
}

# ==================== 【新增】从 CSV 读取并批量调用 API ====================

def main():
    # ---------- 1. 读取 CSV ----------
    csv_file = "./shanghaideal.csv"
    df = pd.read_csv(csv_file)

    if "诉求描述" not in df.columns:
        raise ValueError("CSV 文件中未找到'诉求描述'列")

    # 收集有效文本及其真实行号（CSV 第 2 行 → 输出行号 1）
    valid_rows = []  # list of (display_id, text)
    for idx, row in df.iterrows():
        text = str(row["诉求描述"]).strip()
        if text and text != "nan":
            display_id = idx + 1  # idx=0(CSV第2行) → display_id=1
            valid_rows.append((display_id, text))

    print(f"共 {len(valid_rows)} 条有效文本，开始分类...")

    # ---------- 2. 分批处理，每批 100 条 ----------
    BATCH_SIZE = 50
    all_results = {}  # 存放所有结果 {display_id: category}

    total_batches = (len(valid_rows) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(valid_rows), desc="分类进度", unit="条") as pbar:
        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(valid_rows))
            batch = valid_rows[start:end]

            # 构建发送给 API 的文本块
            batch_text_lines = []
            for display_id, text in batch:
                # 截断过长的文本（避免超 token 限制）
                truncated = text[:800] if len(text) > 800 else text
                batch_text_lines.append(f"{display_id}.{truncated}")

            batch_text = "\n".join(batch_text_lines)

            # ---------- 3. 调用 API ----------
            try:
                response = client.chat.completions.create(
                    model='qwen-plus-2025-07-28',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': batch_text}
                    ],
                    stream=False,
                    extra_body=extra_body
                )

                # 提取返回内容
                reply_content = response.choices[0].message.content.strip()
                print(f"\n--- 批次 {batch_idx + 1}/{total_batches} 返回结果 ---")
                print(reply_content)

                # ---------- 4. 解析返回结果 ----------
                # 格式期望：1：17 或 1:17
                import re
                for line in reply_content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r'(\d+)[:：](\d+)[:：](.+)', line)
                    if match:
                        result_id = int(match.group(1))
                        category = int(match.group(2))
                        group_type = match.group(3).strip()
                        all_results[result_id] = (category, group_type)

            except Exception as e:
                print(f"\n批次 {batch_idx + 1} 调用出错: {e}")
                # 出错时该批次标记为默认值 0
                for display_id, _ in batch:
                    all_results[display_id] = (0, "c")

            pbar.update(len(batch))

    # ---------- 5. 按行号顺序输出最终结果 ----------
    print("\n\n========== 最终分类结果 ==========")
    sorted_ids = sorted(all_results.keys())
    for sid in sorted_ids:
        cat, grp = all_results[sid]
        print(f"{sid}:{cat}:{grp}")

    # ---------- 6. 可选：保存到文件 ----------
    with open("./classification_result.txt", "w", encoding="utf-8") as f:
        for sid in sorted_ids:
            cat, grp = all_results[sid]
            f.write(f"{sid}:{cat}:{grp}\n")
    print(f"\n结果已保存到 ./classification_result.txt")


if __name__ == "__main__":
    main()
