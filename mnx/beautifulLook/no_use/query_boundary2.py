import json
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union

# ========== 读取数据 ==========

# 房山区整体边界
with open('fangshan_geo.json', 'r', encoding='utf-8') as f:
    fangshan_geo = json.load(f)
fangshan_shape = shape(fangshan_geo['features'][0]['geometry'])

# 已有的乡镇边界
with open('fangshan_townships_geo.json', 'r', encoding='utf-8') as f:
    existing_geo = json.load(f)

existing_names = {f['properties']['name'] for f in existing_geo['features']}
print(f"已有边界: {len(existing_names)} 个")

# 28个乡镇的中心坐标（从你的高德API数据中提取）
# 请用你实际的 fangshan.json 中的 center 数据
with open('fangshan.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

township_data = []
for sub in raw['districts'][0]['districts']:
    lng, lat = sub['center'].split(',')
    township_data.append({
        'name': sub['name'],
        'lng': float(lng),
        'lat': float(lat)
    })

# ========== 找出缺失的乡镇 ==========
missing = [t for t in township_data if t['name'] not in existing_names]
print(f"需要补全: {[m['name'] for m in missing]}")

if not missing:
    print("无需补全，退出")
    exit()

# ========== Voronoi 分割 ==========

# 用所有28个中心点做Voronoi
points = np.array([[t['lng'], t['lat']] for t in township_data])

# 添加边界外的辅助点，防止Voronoi区域无限延伸
bounds = fangshan_shape.bounds  # (minx, miny, maxx, maxy)
margin = 0.5
aux_points = [
    [bounds[0] - margin, bounds[1] - margin],
    [bounds[2] + margin, bounds[1] - margin],
    [bounds[0] - margin, bounds[3] + margin],
    [bounds[2] + margin, bounds[3] + margin],
]
all_points = np.vstack([points, aux_points])

vor = Voronoi(all_points)

# 创建一个大的裁剪框
clip_box = box(bounds[0] - margin, bounds[1] - margin,
               bounds[2] + margin, bounds[3] + margin)

# 提取每个原始点对应的Voronoi区域
for i, t in enumerate(township_data):
    if t['name'] not in {m['name'] for m in missing}:
        continue

    region_index = vor.point_region[i]
    region = vor.regions[region_index]

    if -1 in region or len(region) == 0:
        print(f"  ✗ {t['name']}: Voronoi区域无效")
        continue

    # Voronoi区域的顶点坐标
    polygon_coords = [vor.vertices[v] for v in region]
    voronoi_poly = Polygon(polygon_coords)

    # 用房山区边界裁剪
    clipped = voronoi_poly.intersection(fangshan_shape)

    # 再减去已有的乡镇边界
    existing_union = unary_union([
        shape(f['geometry']) for f in existing_geo['features']
    ])
    clipped = clipped.difference(existing_union)

    if clipped.is_empty:
        print(f"  ✗ {t['name']}: 裁剪后为空")
        continue

    # 转为GeoJSON
    if isinstance(clipped, Polygon):
        clipped = MultiPolygon([clipped])

    feature = {
        "type": "Feature",
        "properties": {
            "name": t['name'],
            "note": "近似边界(Voronoi)"
        },
        "geometry": mapping(clipped)
    }
    existing_geo['features'].append(feature)
    print(f"  ✓ {t['name']}: 已生成近似边界")

# ========== 保存最终结果 ==========
with open('fangshan_townships_geo_final.json', 'w', encoding='utf-8') as f:
    json.dump(existing_geo, f, ensure_ascii=False, indent=2)

print(f"\n最终共 {len(existing_geo['features'])} 个乡镇/街道")
print("已保存为 fangshan_townships_geo_final.json")
