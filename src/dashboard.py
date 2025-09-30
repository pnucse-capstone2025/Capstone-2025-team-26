# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = Path(os.getenv("DB_PATH") + "\\agent_activity.db")

st.set_page_config(page_title="AI Refactor Agent Dashboard", layout="wide")

@st.cache_data(ttl=5)
def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        activity = pd.read_sql_query(
            """
            SELECT * 
            FROM activity_log
            ORDER BY timestamp DESC 
            LIMIT 200
            """,
            conn
        )
        issues = pd.read_sql_query(
            """
            SELECT * 
            FROM issue_stats 
            ORDER BY timestamp DESC 
            LIMIT 100
            """, 
            conn
        )
        builds = pd.read_sql_query(
            """
            SELECT * 
            FROM build_log 
            ORDER BY timestamp DESC 
            LIMIT 50
            """, 
            conn
        )
        conn.close()
        return activity, issues, builds
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

st.title("AI 코드 리팩토링 에이전트 대시보드")
activity_df, issue_df, build_df = load_data()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("남은 이슈", str(issue_df.iloc[0]['remaining_issues']) if not issue_df.empty else "0")
with col2:
    if not build_df.empty:
        b = build_df.copy()
        b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
        latest = b.sort_values("timestamp", ascending=False).iloc[0]
        status = str(latest["status"]).strip().lower()
        st.metric("최근 빌드", status)
    else:
        st.metric("최근 빌드", "미실행")
with col3:
    pass
with col4:
    pass

st.markdown("---")
tab1, tab2 = st.tabs(["이슈 추이", "활동 로그"])

with tab1:
    if not issue_df.empty:
        issue_df['timestamp'] = pd.to_datetime(issue_df['timestamp'])
        fig = px.line(issue_df, x="timestamp", y=["remaining_issues"],
                      labels={"value":"#issues","variable":"variable"}, title="이슈 추이")
        st.plotly_chart(fig, use_container_width=True)
        try:
            sev = json.loads(issue_df.iloc[0]['severity_distribution'])
            st.bar_chart(pd.Series(sev, name="count"))
        except Exception:
            pass
    else:
        st.info("이슈 데이터가 없습니다.")

with tab2:
    if not activity_df.empty:
        st.dataframe(activity_df.head(50), use_container_width=True)
    else:
        st.info("활동 로그가 없습니다.")
