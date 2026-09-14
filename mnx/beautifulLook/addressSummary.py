import json

# 目标搜索的乡镇/街道列表
target_locations = [
    "城关街道", "新镇街道", "向阳街道", "东风街道", 
    "迎风街道", "星城街道", "拱辰街道", "西潞街道",
    "良乡镇", "周口店镇", "琉璃河镇", "阎村镇", 
    "窦店镇", "石楼镇", "长阳镇", "河北镇", 
    "长沟镇", "大石窝镇", "张坊镇", "十渡镇", 
    "青龙湖镇", "韩村镇", "霞云岭乡", "南窖乡", 
    "佛子庄乡", "大安山乡", "史家营乡", "蒲洼乡"
]

# 初始化结果字典: 每个目标地点 -> 对应行号列表
result = {loc: [] for loc in target_locations}

# 读取 JSON 文件
with open('../extract/result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每一行
for key in data:
    # 从 "l1" 提取行号 1
    line_num = int(key[1:])
    
    entry = data[key]
    
    # 合并 LOC 和 ORG 列表进行搜索
    all_entities = entry.get("LOC", []) + entry.get("ORG", [])
    
    # 检查每个目标地点是否出现在当前行的实体中
    for loc in target_locations:
        if loc in all_entities:
            result[loc].append(line_num)

map_data = []
for loc in target_locations:
    map_data.append({
        "name": loc,
        "value": len(result[loc])
    })

with open('./fangshan_showtimes.json', 'w', encoding='utf-8') as f:
    json.dump(map_data, f, ensure_ascii=False, indent=2)

print(f"\n已导出地图数据到 fangshan_showtimes.json，共 {len(map_data)} 个乡镇")

# 打印结果
print("=" * 60)
print("房山区乡镇/街道 匹配结果")
print("=" * 60)

line_data = []
for loc in target_locations:
    if result[loc]:
        lines_str = ",".join(map(str, result[loc]))
        count = len(result[loc])
        print(f"{loc} 对应行:{lines_str}【共{count}条】")
        line_data.append({
            "name": loc,
            "lineNum": lines_str
        })
    else:
        print(f"{loc} 对应行:(无匹配)【共0条】")
        line_data.append({
            "name": loc,
            "lineNum": ""
        })
print("=" * 60)

with open('./fangshan_linenum.json', 'w', encoding='utf-8') as f:
    json.dump(line_data, f, ensure_ascii=False, indent=2)

print(f"\n已导出行号数据到 fangshan_linenum.json，共 {len(line_data)} 个乡镇")
