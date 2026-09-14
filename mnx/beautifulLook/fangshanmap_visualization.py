import streamlit as st
from pathlib import Path
import json
import pandas as pd
from mnx.accountLedger.Ledger import load_from_db, CATEGORY_MAP
from streamlit_echarts import st_echarts, Map

def api_fangshanmap_visualization():
    st.subheader("🗺️ 房山区投诉数据地理分布")

    # --- 数据加载 ---
    current_dir = Path(__file__).resolve().parent
    map_data_path = current_dir / 'fangshan.json'
    showtimes_path = current_dir / 'fangshan_showtimes.json'
    linenum_path = current_dir / 'fangshan_linenum.json'

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

    township_line_map = {item['name']: item['lineNum'] for item in line_num_data}
    max_value = max([item['value'] for item in stat_data]) if stat_data else 1

    # --- 初始化 session_state ---
    if "selected_township" not in st.session_state:
        st.session_state.selected_township = None

    # --- 注册地图 ---
    fangshan_map = Map("FANGSHAN", geo_json_data)

    # --- ECharts 配置 ---
    options = {
        "backgroundColor": "transparent",
        "title": {
            "text": "房山区乡镇/街道投诉分布",
            "left": "center",
            "textStyle": {"color": "#fff", "fontSize": 18}
        },
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}<br/>投诉数量: {c}条"
        },
        "visualMap": {
            "min": 0,
            "max": max_value,
            "left": "left",
            "top": "bottom",
            "text": ["高", "低"],
            "inRange": {
                "color": ["#323c48", "#50a3ba", "#eac736", "#d94e5d", "#ff4500"]
            },
            "textStyle": {"color": "#fff"},
            "calculable": True
        },
        "series": [
            {
                "name": "投诉数据",
                "type": "map",
                "map": "FANGSHAN",
                "selectedMode": False,
                "roam": True,
                "label": {"show": True, "color": "#fff", "fontSize": 10},
                "itemStyle": {
                    "areaColor": "#323c48",
                    "borderColor": "#111",
                    "borderWidth": 0.5
                },
                "emphasis": {
                    "label": {"color": "#fff", "fontSize": 12},
                    "itemStyle": {"areaColor": "#2a333d"}
                },
                "data": stat_data
            }
        ],
    }

    # =============================================
    # 使用 streamlit-echarts 并处理点击事件
    # =============================================
    res = st_echarts(
        options=options,
        map=fangshan_map,
        height="600px",
        key="fs_map_v2"
    )

    # --- 下方表格显示逻辑 ---
    st.markdown("---")

    selected_township = st.session_state.get("selected_township")

    # 布局：左侧标题 + 右侧按钮
    header_col, clear_col, select_col = st.columns([3, 1, 2])
    with header_col:
        st.subheader(f"📊 {selected_township if selected_township else '全部地区'} 数据统计")
    
    with clear_col:
        if selected_township:
            if st.button("🔄 清除选择", use_container_width=True, key="clear_btn"):
                st.session_state.selected_township = None
                st.rerun()
    
    with select_col:
        # 备用下拉选择器
        township_names = list(township_line_map.keys())
        current_index = 0
        if selected_township and selected_township in township_names:
            current_index = township_names.index(selected_township) + 1

        selected_manual = st.selectbox(
            "📍 手动选择乡镇/街道",
            ["请选择..."] + township_names,
            index=current_index,
            key="township_selector_v2",
            label_visibility="collapsed"
        )

    # --- 确定最终选中的乡镇 ---
    final_township = None
    
    # 优先使用手动选择
    if selected_manual and selected_manual != "请选择...":
        if selected_manual != selected_township:
            final_township = selected_manual
            st.session_state.selected_township = final_township
            st.rerun()  # 触发rerun以确保state同步
        else:
            final_township = selected_manual
    # # 其次使用地图点击选择
    # elif selected_township:
    #     final_township = selected_township

    # --- 显示数据表格 ---
    if final_township:
        selected_lines = township_line_map.get(final_township)
        if selected_lines:
            try:
                line_numbers = [int(x.strip()) for x in selected_lines.split(',')]
            except Exception as e:
                st.error(f"❌ 解析行号失败: {e}")
                return

            db_data = load_from_db()
            if not db_data.empty:
                filtered_df = db_data[db_data['id'].isin(line_numbers)]
                if not filtered_df.empty:
                    # 映射预测类别
                    if "pred_category" in filtered_df.columns:
                        filtered_df["预测类别"] = filtered_df["pred_category"].map(CATEGORY_MAP)
                    else:
                        filtered_df["预测类别"] = "未分类"
                    
                    # 确保所有需要的列都是字符串类型
                    display_columns = ["content", "预测类别", "manual_review", "status", "create_time"]
                    for col in display_columns:
                        if col in filtered_df.columns:
                            filtered_df[col] = filtered_df[col].astype(str)
                        else:
                            filtered_df[col] = "无数据"

                    st.info(f"📍 **{final_township}** — 共 {len(filtered_df)} 条投诉数据")

                    st.dataframe(
                        filtered_df.set_index('id')[display_columns],
                        use_container_width=True,
                        column_config={
                            "content": st.column_config.TextColumn("投诉内容", width="large"),
                            "预测类别": st.column_config.TextColumn("预测类别", width="medium"),
                            "manual_review": st.column_config.TextColumn("人工复核", width="small"),
                            "status": st.column_config.TextColumn("系统状态", width="small"),
                            "create_time": st.column_config.TextColumn("入库时间", width="medium"),
                        }
                    )
                else:
                    st.warning(f"⚠️ **{final_township}** 对应行号在数据库中未找到数据")
            else:
                st.warning("⚠️ 数据库暂无数据")
        else:
            st.warning(f"⚠️ 未找到 **{final_township}** 的行号配置")
    else:
        st.info("💡 点击地图上的乡镇，或使用上方下拉框，查看该区域的投诉数据")


