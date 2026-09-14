import requests
import json
import time

# 28个乡镇/街道
townships = [
    "佛子庄乡", "蒲洼乡", "南窖乡", "史家营乡", "大安山乡",
    "窦店镇", "良乡镇", "霞云岭乡", "长沟镇", "琉璃河镇",
    "新镇街道", "韩村河镇", "十渡镇", "张坊镇", "石楼镇",
    "河北镇", "青龙湖镇", "周口店镇", "城关街道", "阎村镇",
    "拱辰街道", "西潞街道", "星城街道", "东风街道", "迎风街道",
    "长阳镇", "大石窝镇", "向阳街道"
]

features = []
failed = []

for i, name in enumerate(townships):
    print(f"[{i+1}/{len(townships)}] 查询: {name} ...", end=" ")

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{name}, 房山区, 北京",
                "format": "json",
                "polygon_geojson": 1,
                "limit": 1
            },
            headers={"User-Agent": "FangshanMapProject/1.0"},  # 必须设置
            timeout=15
        )

        results = resp.json()

        if results and 'geojson' in results[0]:
            geo = results[0]['geojson']

            # 确保是面数据（Polygon或MultiPolygon），排除Point
            if geo['type'] in ['Polygon', 'MultiPolygon']:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "osm_type": results[0].get('osm_type', ''),
                        "osm_id": results[0].get('osm_id', '')
                    },
                    "geometry": geo
                }
                features.append(feature)
                print(f"✓ ({geo['type']})")
            else:
                print(f"✗ (返回的是{geo['type']}，不是面数据)")
                failed.append(name)
        else:
            print("✗ (无结果)")
            failed.append(name)

    except Exception as e:
        print(f"✗ (错误: {e})")
        failed.append(name)

    time.sleep(1.1)  # Nominatim限制：每秒1次请求

# 保存结果
geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open('fangshan_townships_geo.json', 'w', encoding='utf-8') as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print(f"\n{'='*40}")
print(f"成功: {len(features)}/{len(townships)}")
print(f"失败: {len(failed)}")
if failed:
    print(f"失败列表: {failed}")
print(f"已保存为 fangshan_townships_geo.json")
