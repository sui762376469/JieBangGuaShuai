import streamlit as st
from pathlib import Path
import json
import pandas as pd
from mnx.accountLedger.Ledger import load_from_db, CATEGORY_MAP

# 引入 streamlit-js-eval 用于获取前端 JavaScript 交互数据
try:
    from streamlit_js_eval import streamlit_js_eval
except ImportError:
    streamlit_js_eval = None


def api_fangshanmap_visualization():
    """
    渲染房山区地图可视化界面
    基于 ECharts 实现，支持点击地图区域查看对应乡镇的投诉数据
    """
    st.subheader("🗺️ 房山区投诉数据地理分布")
    
    # 获取项目根目录
    current_dir = Path(__file__).resolve().parent
    map_data_path = current_dir / 'fangshan.json'
    showtimes_path = current_dir / 'fangshan_showtimes.json'
    linenum_path = current_dir / 'fangshan_linenum.json'
    
    # 检查文件是否存在
    if not map_data_path.exists() or not showtimes_path.exists() or not linenum_path.exists():
        st.error("❌ 地图数据文件不存在，请检查路径配置")
        return
    
    # 加载地图GeoJSON数据和统计数据
    try:
        with open(map_data_path, 'r', encoding='utf-8') as f:
            geo_json_data = json.load(f)
        with open(showtimes_path, 'r', encoding='utf-8') as f:
            stat_data = json.load(f)
        with open(linenum_path, 'r', encoding='utf-8') as f:
            line_num_data = json.load(f)
    except Exception as e:
        st.error(f"❌ 加载数据失败: {e}")
        return
    
    # 初始化session_state用于存储选中的乡镇
    if "selected_township" not in st.session_state:
        st.session_state.selected_township = None
    
    # 计算最大值用于视觉映射
    max_value = max([item['value'] for item in stat_data]) if stat_data else 1
    
    # 构建乡镇名称到行号的映射字典
    township_line_map = {item['name']: item['lineNum'] for item in line_num_data}
    
    # ========== 关键修改：使用 sessionStorage 在 iframe 和主页面之间传递数据 ==========
    # 先读取 sessionStorage 中是否有点击的乡镇（每次页面加载时检查）
    if streamlit_js_eval is not None:
        try:
            stored_township = streamlit_js_eval(
                js_expressions="sessionStorage.getItem('clicked_township')",
                key="read_session_storage"
            )
            if stored_township and isinstance(stored_township, str) and stored_township != st.session_state.selected_township:
                st.session_state.selected_township = stored_township
                # 读取后清除，避免重复触发
                streamlit_js_eval(
                    js_expressions="sessionStorage.removeItem('clicked_township')",
                    key="clear_session_storage"
                )
                st.rerun()
        except Exception:
            pass
    
    # 构建 ECharts HTML，将GeoJSON数据直接嵌入
    # 关键修改：点击地图时，将乡镇名称写入 sessionStorage，然后刷新父页面
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
            var townshipLineMap = {json.dumps(township_line_map, ensure_ascii=False)};
            
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
            
            // 添加点击事件监听
            myChart.on('click', function(params) {{
                var townshipName = params.name;
                console.log('Clicked township:', townshipName);
                
                // 将选中的乡镇名称写入 sessionStorage（与主页面共享）
                try {{
                    sessionStorage.setItem('clicked_township', townshipName);
                    // 刷新父页面，让 Streamlit 重新运行并读取 sessionStorage
                    window.parent.location.reload();
                }} catch(e) {{
                    console.log('sessionStorage failed:', e);
                }}
            }});
            
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
    
    # 显示下属地区数据统计标题
    st.subheader("下属地区数据统计")
    
    # 创建刷新按钮和当前选中提示
    refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 1, 4])
    with refresh_col1:
        if st.button("🔄 清除选择", key="clear_township_selection"):
            st.session_state.selected_township = None
            # 清除 sessionStorage
            if streamlit_js_eval is not None:
                try:
                    streamlit_js_eval(
                        js_expressions="sessionStorage.removeItem('clicked_township')",
                        key="clear_storage"
                    )
                except Exception:
                    pass
            st.rerun()
    with refresh_col2:
        if st.button("🔄 刷新表格", key="refresh_table_btn"):
            st.rerun()
    
    # 由于URL hash检测在Streamlit中有限制，我们提供一个备用的下拉选择器
    township_names = list(township_line_map.keys())
    
    select_col1, select_col2 = st.columns([3, 1])
    with select_col1:
        # 将 session_state 中的值同步到下拉框
        current_index = 0
        if st.session_state.selected_township and st.session_state.selected_township in township_names:
            current_index = township_names.index(st.session_state.selected_township) + 1
            
        selected_township_manual = st.selectbox(
            "📍 或手动选择乡镇/街道查看详细信息",
            ["请选择..."] + township_names,
            index=current_index,
            key="township_selector_manual",
            help="如果点击地图未生效，可使用此下拉框选择"
        )
    with select_col2:
        st.write("")  # 占位符保持布局
    
    # 确定最终选中的乡镇（优先使用手动选择，其次使用session_state）
    final_selected_township = None
    if selected_township_manual and selected_township_manual != "请选择...":
        final_selected_township = selected_township_manual
        # 同步更新 session_state
        if st.session_state.selected_township != final_selected_township:
            st.session_state.selected_township = final_selected_township
    elif st.session_state.selected_township:
        final_selected_township = st.session_state.selected_township
    
    # 根据选中的乡镇显示对应的数据
    if final_selected_township:
        # 从映射中找到对应的行号
        selected_lines = township_line_map.get(final_selected_township)
        
        if selected_lines:
            # 解析行号
            try:
                line_numbers = [int(x.strip()) for x in selected_lines.split(',')]
            except Exception as e:
                st.error(f"❌ 解析行号失败: {e}")
                return
            
            # 从数据库加载数据
            db_data = load_from_db()
            
            if not db_data.empty:
                # 根据行号筛选数据（数据库id从1开始，行号也是从1开始）
                filtered_df = db_data[db_data['id'].isin(line_numbers)]
                
                if not filtered_df.empty:
                    filtered_df["预测类别"] = filtered_df["pred_category"].map(CATEGORY_MAP)
                    for col in ["content", "预测类别", "manual_review", "status", "create_time"]:
                        filtered_df[col] = filtered_df[col].astype(str)
                    
                    st.info(f"📍 当前显示：**{final_selected_township}** 的 {len(filtered_df)} 条投诉数据（行号: {selected_lines}）")
                    
                    display_df = filtered_df.set_index('id')
                    
                    st.dataframe(
                        display_df[["content", "预测类别", "manual_review", "status", "create_time"]],
                        use_container_width=True,
                        column_config={
                            "content": st.column_config.TextColumn("投诉内容", width="large"),
                            "预测类别": st.column_config.TextColumn("预测类别", width="medium"),
                            "manual_review": st.column_config.TextColumn("人工复核", width="small"),
                            "status": st.column_config.TextColumn("系统状态", width="small"),
                            "create_time": st.column_config.TextColumn("入库时间", width="medium")
                        }
                    )
                else:
                    st.warning(f"⚠️ **{final_selected_township}** 对应的行号在数据库中未找到数据（行号: {selected_lines}）")
            else:
                st.warning("⚠️ 数据库暂无数据，请先在【数据导入与预测】标签页导入数据")
        else:
            st.warning(f"⚠️ 未找到 **{final_selected_township}** 的行号配置")
    else:
        st.info("💡 **点击地图上的乡镇/街道**，或使用下方下拉框，查看该区域的详细投诉数据")