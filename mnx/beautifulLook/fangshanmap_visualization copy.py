import streamlit as st
from pathlib import Path
import json


def api_fangshanmap_visualization():
    """
    渲染房山区地图可视化界面
    基于 ECharts 实现，只显示地图部分，不包含右侧面板和拖拽分隔条
    """
    st.subheader("🗺️ 房山区投诉数据地理分布")
    
    # 获取项目根目录
    current_dir = Path(__file__).resolve().parent
    map_data_path = current_dir / 'fangshan.json'
    showtimes_path = current_dir / 'fangshan_showtimes.json'
    
    # 检查文件是否存在
    if not map_data_path.exists() or not showtimes_path.exists():
        st.error("❌ 地图数据文件不存在，请检查路径配置")
        return
    
    # 加载地图GeoJSON数据和统计数据
    try:
        with open(map_data_path, 'r', encoding='utf-8') as f:
            geo_json_data = json.load(f)
        with open(showtimes_path, 'r', encoding='utf-8') as f:
            stat_data = json.load(f)
    except Exception as e:
        st.error(f"❌ 加载数据失败: {e}")
        return
    
    # 计算最大值用于视觉映射
    max_value = max([item['value'] for item in stat_data]) if stat_data else 1
    
    # 构建 ECharts HTML，将GeoJSON数据直接嵌入
    echarts_html = f"""
    <!DOCTYPE html>
    <html style="height: 100%">
    <head>
        <meta charset="utf-8">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            html, body {{
                height: 100%;
                overflow: hidden;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            
            #container {{
                width: 100%;
                height: 600px;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        
        <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
        <script>
            var myChart = echarts.init(document.getElementById("container"));
            var mapData = {json.dumps(stat_data, ensure_ascii=False)};
            var geoJson = {json.dumps(geo_json_data, ensure_ascii=False)};
            
            // 直接注册嵌入的地图数据
            echarts.registerMap('fangshan', geoJson);
            
            var option = {{
                backgroundColor: '#404a59',
                title: {{
                    text: '房山区乡镇/街道投诉分布',
                    left: 'center',
                    textStyle: {{ color: '#fff', fontSize: 18 }}
                }},
                tooltip: {{
                    trigger: 'item',
                    formatter: function(params) {{
                        return params.name + '<br/>投诉数量: ' + (params.value || 0) + '条';
                    }}
                }},
                visualMap: {{
                    min: 0,
                    max: {max_value},
                    left: 'left',
                    top: 'bottom',
                    text: ['高', '低'],
                    inRange: {{
                        color: ['#323c48', '#50a3ba', '#eac736', '#d94e5d', '#ff4500']
                    }},
                    textStyle: {{ color: '#fff' }},
                    calculable: true
                }},
                series: [{{
                    name: '投诉数据',
                    type: 'map',
                    map: 'fangshan',
                    selectedMode: false,
                    roam: true,
                    label: {{
                        show: true,
                        color: '#fff',
                        fontSize: 10
                    }},
                    itemStyle: {{
                        areaColor: '#323c48',
                        borderColor: '#111',
                        borderWidth: 0.5
                    }},
                    emphasis: {{
                        label: {{ color: '#fff', fontSize: 12 }},
                        itemStyle: {{ areaColor: '#2a333d' }}
                    }},
                    data: mapData
                }}]
            }};
            
            myChart.setOption(option);
            
            // 响应式调整
            window.addEventListener('resize', function() {{
                myChart.resize();
            }});
        </script>
    </body>
    </html>
    """
    
    # 在 Streamlit 中渲染 HTML
    st.components.v1.html(echarts_html, height=650, scrolling=False)
    st.subheader("下属地区数据统计")
