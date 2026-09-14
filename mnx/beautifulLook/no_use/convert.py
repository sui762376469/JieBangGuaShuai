import json

# 读取高德API返回的原始数据
with open('fangshan.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

district = data['districts'][0]

# ========== 1. 转换房山区整体边界 ==========
polyline_str = district['polyline']
# 高德格式: "lng1,lat1;lng2,lat2;..."  多个面用 "|" 分隔
polygons_str = polyline_str.split('|')

coordinates = []
for poly in polygons_str:
    points = poly.strip().split(';')
    ring = []
    for p in points:
        if ',' in p:
            lng, lat = p.split(',')
            ring.append([float(lng), float(lat)])
    coordinates.append(ring)

geojson = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"name": "房山区", "adcode": "110111"},
        "geometry": {
            "type": "Polygon" if len(coordinates) == 1 else "MultiPolygon",
            "coordinates": coordinates if len(coordinates) == 1 else [coordinates]
        }
    }]
}

with open('fangshan_geo.json', 'w', encoding='utf-8') as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)
print("✓ 已生成 fangshan_geo.json（房山区边界）")

# ========== 2. 提取各乡镇中心坐标 ==========
townships = []
for sub in district['districts']:
    lng, lat = sub['center'].split(',')
    townships.append({
        'name': sub['name'],
        'lng': float(lng),
        'lat': float(lat)
    })

with open('township_centers.json', 'w', encoding='utf-8') as f:
    json.dump(townships, f, ensure_ascii=False, indent=2)
print(f"✓ 已生成 township_centers.json（{len(townships)}个乡镇/街道中心点）")

# 打印所有乡镇名称供确认
print("\n乡镇/街道列表：")
for t in townships:
    print(f"  {t['name']}: [{t['lng']}, {t['lat']}]")
