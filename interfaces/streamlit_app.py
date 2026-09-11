"""
interfaces/streamlit_app.py - Streamlit HITL Approval & Monitoring UI.
Thin client consuming interfaces/api_server.py.
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Coding Agent Harness", page_icon="🤖", layout="wide")

st.title("🤖 Autonomous Coding Agent Harness")
st.caption("Execution harness with Human-In-The-Loop approval gate")

# Sidebar for session management
with st.sidebar:
    st.header("Session Settings")
    thread_id = st.text_input("Thread ID", value="default-build-thread")
    api_url = st.text_input("API URL", value=API_BASE)

# Main tabs
tab_run, tab_approvals, tab_logs = st.tabs(["🚀 Run Agent", "🛡️ Approvals (HITL)", "📊 Progress & Logs"])

with tab_run:
    st.subheader("Start New Run")
    user_prompt = st.text_area(
        "Software Build Request",
        placeholder="e.g. Build a FastAPI service with SQLite auth and tests.",
        height=150,
    )
    if st.button("Submit Build Task", type="primary"):
        if not user_prompt.strip():
            st.warning("Please provide a build request.")
        else:
            with st.spinner("Submitting task to harness..."):
                try:
                    resp = requests.post(
                        f"{api_url}/runs",
                        json={"user_request": user_prompt, "thread_id": thread_id},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Run status: {data.get('status')}")
                        st.json(data)
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")

with tab_approvals:
    st.subheader("Pending HITL Approvals")
    st.info("When an action matches an 'ask' rule (e.g. git push, deploy, file write), it pauses here for review.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Action", use_container_width=True, type="primary"):
            try:
                resp = requests.post(
                    f"{api_url}/runs/{thread_id}/resume",
                    json={"approved": True},
                    timeout=60,
                )
                st.success("Approval sent!")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed: {e}")

    with col2:
        if st.button("❌ Reject Action", use_container_width=True):
            try:
                resp = requests.post(
                    f"{api_url}/runs/{thread_id}/resume",
                    json={"approved": False},
                    timeout=60,
                )
                st.warning("Rejection sent!")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Failed: {e}")

with tab_logs:
    st.subheader("Thread State & Artifacts")
    if st.button("Refresh State"):
        try:
            resp = requests.get(f"{api_url}/runs/{thread_id}")
            if resp.status_code == 200:
                st.json(resp.json())
            else:
                st.info("No active state found for this thread.")
        except Exception as e:
            st.error(f"Could not load state: {e}")
