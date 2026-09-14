import json
import difflib
import re
import pandas as pd
from collections import defaultdict


def calculate_similarity(text1, text2):
    """计算字面相似度"""
    if not text1 and not text2: return 1.0
    if not text1 or not text2: return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def get_entity_clusters(entity_list, threshold=0.7):
    """聚类算法：对去重后的实体列表进行相似度聚类"""
    nodes = list(entity_list)
    graph = {node: [] for node in nodes}

    # 两两计算实体相似度
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            # with open('similarity_debug.txt', 'a', encoding='utf-8') as f:
            #     f.write(f"Comparing: {u} vs {v}\n")
            #     f.write(f"Similarity: {calculate_similarity(u, v)}\n\n")
            if calculate_similarity(u, v) >= threshold:
                graph[u].append(v)
                graph[v].append(u)

    visited = set()
    clusters = []
    for node in nodes:
        if node not in visited:
            component = []
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    component.append(curr)
                    stack.extend(graph[curr])
            clusters.append(component)
    return clusters


def natural_sort_key(s):
    """自然排序辅助函数：确保 l2 排在 l10 前面"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def is_valid_entity(text):
    """
    【升级版拦截器】判断实体是否为有效的具体名称。
    彻底过滤中英文脱敏符号。
    """
    if not text:
        return False

    # 规则 1：拦截常见脱敏符号
    # 包含了：数学乘号(×), 星号(*), 井号(#), 全角星号(＊)
    # 【新增】：英文大写(X), 英文小写(x), 全角大写(Ｘ), 全角小写(ｘ)
    if re.search(r'[×*#＊XxＸｘ]', text):
        return False

    # 规则 2：拦截“某”字占位符（如“某某路”、“王某”）
    if '某' in text:
        return False

    # 规则 3：如果提取出来的实体只有一个字（如只有个“路”或“局”字），通常也是提取噪音，直接过滤
    if len(text.strip()) <= 1:
        return False

    return True


def process_and_export_csv(json_path, loc_csv_name, per_csv_name):
    # 1. 加载数据
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"未找到 {json_path}，请检查文件路径。")
        return

    per_to_rows = defaultdict(list)
    loc_org_to_rows = defaultdict(list)

    # 2. 倒排索引提取
    for row_id, content in data.items():
        # -- 提取人名 --
        for per in content.get("PER", []):
            clean_per = per.strip()
            # 必须通过有效性校验
            if is_valid_entity(clean_per):
                per_to_rows[clean_per].append(row_id)

        # -- 提取地点和机构 --
        loc_str = "".join(content.get("LOC", []))
        org_str = "".join(content.get("ORG", []))
        loc_org_combined = (loc_str + org_str).strip()

        # 必须通过有效性校验
        if is_valid_entity(loc_org_combined):
            loc_org_to_rows[loc_org_combined].append(row_id)

    # 3. 执行聚类分析
    print("正在分析人名相似度...")
    per_clusters = get_entity_clusters(per_to_rows.keys(), threshold=0.85)

    print("正在分析点位相似度...")
    loc_clusters = get_entity_clusters(loc_org_to_rows.keys(), threshold=0.75)

    # 4. 构造人名结果
    per_res = []
    for cluster in per_clusters:
        associated_rows = set()
        for name in cluster:
            associated_rows.update(per_to_rows[name])

        sorted_rows = sorted(list(associated_rows), key=natural_sort_key)

        per_res.append({
            "标准姓名": cluster[0],
            "包含的相似变体": " | ".join(cluster),
            "关联投诉次数": len(sorted_rows),
            "具体行号": ", ".join(sorted_rows)
        })

    df_per = pd.DataFrame(per_res)
    if not df_per.empty:
        df_per.sort_values(by="关联投诉次数", ascending=False).to_csv(per_csv_name, index=False, encoding='utf-8-sig')
        print(f"人名统计已导出至: {per_csv_name}")

    # 5. 构造点位结果
    loc_res = []
    for cluster in loc_clusters:
        associated_rows = set()
        for loc in cluster:
            associated_rows.update(loc_org_to_rows[loc])

        sorted_rows = sorted(list(associated_rows), key=natural_sort_key)

        loc_res.append({
            "标准点位": cluster[0],
            "包含的相似变体": " | ".join(cluster),
            "关联投诉次数": len(sorted_rows),
            "具体行号": ", ".join(sorted_rows)
        })

    df_loc = pd.DataFrame(loc_res)
    if not df_loc.empty:
        df_loc.sort_values(by="关联投诉次数", ascending=False).to_csv(loc_csv_name, index=False, encoding='utf-8-sig')
        print(f"点位统计已导出至: {loc_csv_name}")

def multi2one(json_path):
    """
    多对一分析核心引擎 - 纯内存版本，不写CSV
    
    Args:
        json_path: result.json 文件路径
        
    Returns:
        row_to_loc_map: dict {row_id_str -> standard_location_name}
        cluster_summary: list[dict] 聚类摘要信息（用于侧边栏展示）
    """
    # 1. 加载 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 倒排索引提取（仅 LOC+ORG）
    loc_org_to_rows = defaultdict(list)
    for row_id, content in data.items():
        loc_str = "".join(content.get("LOC", []))
        org_str = "".join(content.get("ORG", []))
        loc_org_combined = (loc_str + org_str).strip()
        
        if is_valid_entity(loc_org_combined):
            loc_org_to_rows[loc_org_combined].append(row_id)

    if not loc_org_to_rows:
        return {}, []

    # 3. 聚类分析
    print(f"[多对一] 共 {len(loc_org_to_rows)} 个唯一点位，开始聚类...")
    loc_clusters = get_entity_clusters(loc_org_to_rows.keys(), threshold=0.75)
    print(f"[多对一] 聚类完成，生成 {len(loc_clusters)} 个标准点位")

    # 4. 构建 {row_id -> 标准点位} 映射
    row_to_loc_map = {}
    cluster_summary = []
    
    for cluster in loc_clusters:
        standard_loc = cluster[0]
        associated_rows = set()
        for loc in cluster:
            associated_rows.update(loc_org_to_rows[loc])
        
        for row_id in associated_rows:
            row_to_loc_map[row_id] = standard_loc
        
        cluster_summary.append({
            "standard_loc": standard_loc,
            "variants": cluster,
            "row_count": len(associated_rows),
            "row_ids": sorted(list(associated_rows), key=natural_sort_key)
        })

    # 按 row_count 降序排列
    cluster_summary.sort(key=lambda x: x["row_count"], reverse=True)
    
    return row_to_loc_map, cluster_summary

def one2multi(json_path):
    """
    一对多人名分析核心引擎 - 纯内存版本
    
    基于result.json中的PER(人名)实体进行相似度聚类，
    识别同一人的不同表述变体（如""/"李先生"等）。
    
    Args:
        json_path: result.json 文件路径
        
    Returns:
        row_to_name_map: dict {row_id_str -> standard_person_name}
        person_summary: list[dict] 人名聚类摘要
            [{name, variants, row_count, row_ids}, ...]
    """
    # 1. 加载 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 倒排索引提取（仅 PER 人名）
    per_to_rows = defaultdict(list)
    for row_id, content in data.items():
        for per in content.get("PER", []):
            clean_per = per.strip()
            if is_valid_entity(clean_per):
                per_to_rows[clean_per].append(row_id)

    if not per_to_rows:
        return {}, []

    # 3. 执行人名聚类分析（阈值比地点稍高，因为人名更敏感）
    print(f"[一对多] 共 {len(per_to_rows)} 个唯一人名，开始聚类...")
    per_clusters = get_entity_clusters(per_to_rows.keys(), threshold=0.85)
    print(f"[一对多] 聚类完成，生成 {len(per_clusters)} 个标准姓名")

    # 4. 构建 {row_id -> 标准姓名} 映射 + 聚类摘要
    row_to_name_map = {}
    person_summary = []
    
    for cluster in per_clusters:
        standard_name = cluster[0]
        associated_rows = set()
        for name in cluster:
            associated_rows.update(per_to_rows[name])
        
        for row_id in associated_rows:
            row_to_name_map[row_id] = standard_name
        
        person_summary.append({
            "standard_name": standard_name,
            "variants": cluster,
            "row_count": len(associated_rows),
            "row_ids": sorted(list(associated_rows), key=natural_sort_key)
        })

    # 按 row_count 降序排列
    person_summary.sort(key=lambda x: x["row_count"], reverse=True)
    
    return row_to_name_map, person_summary

if __name__ == "__main__":
    input_json = "result.json"
    output_loc_csv = "点位分布统计.csv"
    output_per_csv = "人员投诉统计.csv"

    process_and_export_csv(input_json, output_loc_csv, output_per_csv)
    print("\n清理完成！请检查最新的 CSV 文件。")