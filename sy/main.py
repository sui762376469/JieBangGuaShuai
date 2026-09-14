import streamlit as st
import importlib
import sys

query_params = st.query_params

# 初始化会话状态
if "page" not in st.session_state:
    st.session_state.page = query_params.get("page", "login")
if "logged_in" not in st.session_state:
    st.session_state.logged_in = query_params.get("logged_in") == "true"
if "user_type" not in st.session_state:
    ut = query_params.get("user_type")
    st.session_state.user_type = ut if ut else None
if "login_mode" not in st.session_state:
    st.session_state.login_mode = query_params.get("login_mode", "user")

def sync_to_url():
    """把当前 session_state 中的关键状态同步到 URL"""
    qp = st.query_params
    if st.session_state.get("page"):
        qp["page"] = st.session_state.page
    if st.session_state.get("logged_in"):
        qp["logged_in"] = "true"
    else:
        qp["logged_in"] = "false"
    if st.session_state.get("user_type"):
        qp["user_type"] = st.session_state.user_type
    if st.session_state.get("login_mode"):
        qp["login_mode"] = st.session_state.login_mode

sync_to_url()
# 动态加载页面
if st.session_state.page == "login":
    exec(open("login.py", "r", encoding="utf-8").read())
elif st.session_state.page == "app":
    import app
    importlib.reload(app)
    app.show_app()
