import streamlit as st
import json
from datetime import datetime
from frontend.utils import api_get, api_post, api_delete

def show_evaluation():
    st.markdown("<h2 style='font-weight: 700; margin-bottom: 20px;'>🧪 Evaluation Lab</h2>", unsafe_allow_html=True)

    # Custom CSS
    st.markdown("""
    <style>
        .eval-card {
            background: linear-gradient(135deg, #1f1f2e 0%, #151522 100%);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #33334d;
            margin-bottom: 16px;
        }
        .metric-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            margin: 2px 4px;
        }
        .metric-good { background: #065f46; color: #6ee7b7; }
        .metric-ok { background: #78350f; color: #fcd34d; }
        .metric-bad { background: #7f1d1d; color: #fca5a5; }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Run Evaluation", "📁 Past Reports"])

    # ── TAB 1: Run Evaluation ──
    with tab1:
        st.markdown("<div class='eval-card'>", unsafe_allow_html=True)
        st.markdown("### Run RAGAS Evaluation")
        st.markdown(
            "Enter question-answer pairs with context and ground truth to evaluate your RAG pipeline. "
            "Requires `OPENAI_API_KEY` to be set on the backend."
        )

        report_name = st.text_input("Report Name", value=f"Eval Run {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

        # Judge model selection
        col1, col2 = st.columns(2)
        with col1:
            judge_model = st.selectbox(
                "Judge LLM Model",
                options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                index=0,
            )
        with col2:
            embedding_model = st.selectbox(
                "Embedding Model",
                options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                index=0,
            )

        # Dataset input
        st.markdown("#### Evaluation Samples")
        st.markdown("JSONL format: each line is `{\"question\": ..., \"answer\": ..., \"contexts\": [...], \"ground_truth\": ...}`")

        input_method = st.radio("Input method", ["Paste JSONL", "Upload JSONL file", "Use sample dataset"], horizontal=True)

        samples_data = []

        if input_method == "Use sample dataset":
            try:
                with open("evaluation/sample_eval_dataset.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples_data.append(json.loads(line))
                st.success(f"Loaded {len(samples_data)} samples from the sample dataset.")
            except Exception as e:
                st.error(f"Could not load sample dataset: {e}")

        elif input_method == "Paste JSONL":
            jsonl_text = st.text_area(
                "Paste JSONL data (one JSON object per line)",
                height=200,
                placeholder='{"question":"...","answer":"...","contexts":["..."],"ground_truth":"..."}',
            )
            if jsonl_text:
                for line_no, line in enumerate(jsonl_text.strip().split("\n"), start=1):
                    line = line.strip()
                    if line:
                        try:
                            samples_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            st.warning(f"Skipping invalid JSON on line {line_no}")

        elif input_method == "Upload JSONL file":
            uploaded_file = st.file_uploader("Upload JSONL file", type=["jsonl", "json"])
            if uploaded_file:
                content = uploaded_file.read().decode("utf-8")
                for line_no, line in enumerate(content.strip().split("\n"), start=1):
                    line = line.strip()
                    if line:
                        try:
                            samples_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            st.warning(f"Skipping invalid JSON on line {line_no}")

        if samples_data:
            st.info(f"Ready: {len(samples_data)} samples loaded.")

            if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
                payload = {
                    "report_name": report_name,
                    "samples": [
                        {
                            "question": s["question"],
                            "answer": s["answer"],
                            "contexts": s.get("contexts", []),
                            "ground_truth": s["ground_truth"],
                        }
                        for s in samples_data
                    ],
                    "judge_model": judge_model,
                    "embedding_model": embedding_model,
                }

                with st.spinner("Running RAGAS evaluation... (this may take a while)"):
                    result = api_post("/evaluate", payload)

                if result:
                    st.success("Evaluation completed!")

                    # Show summary
                    summary = result.get("summary", {})
                    st.markdown("#### Aggregate Scores")
                    cols = st.columns(4)
                    metric_labels = {
                        "faithfulness": "Faithfulness",
                        "answer_relevancy": "Answer Relevancy",
                        "context_precision": "Context Precision",
                        "context_recall": "Context Recall",
                    }
                    for i, (metric_key, metric_label) in enumerate(metric_labels.items()):
                        value = summary.get(metric_key)
                        if value is not None:
                            with cols[i]:
                                color_class = "metric-good" if value >= 0.7 else ("metric-ok" if value >= 0.4 else "metric-bad")
                                st.markdown(
                                    f"<div style='text-align:center;'>"
                                    f"<div style='font-size:13px;color:#a0a0c0;'>{metric_label}</div>"
                                    f"<div class='metric-badge {color_class}' style='font-size:18px;padding:6px 16px;'>{value:.4f}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.metric("Total Samples", summary.get("rows", 0))
                    st.json(result)
                else:
                    st.error("Evaluation failed. Check that OPENAI_API_KEY is set on the backend and try again.")
        else:
            st.info("Load or paste evaluation data to get started.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Past Reports ──
    with tab2:
        st.markdown("<div class='eval-card'>", unsafe_allow_html=True)
        st.markdown("### Past Evaluation Reports")

        reports = api_get("/evaluate/reports")

        if reports is None:
            st.warning("Could not fetch reports from backend. Is the server running?")
        elif not reports:
            st.info("No evaluation reports found. Run an evaluation in the tab above.")
        else:
            for report in reports:
                report_id = report["id"]
                report_name = report.get("report_name", "Unnamed")
                dataset_size = report.get("dataset_size", 0)
                created_raw = report.get("created_at", "")
                try:
                    dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    created_str = dt.strftime("%b %d, %Y - %I:%M %p")
                except Exception:
                    created_str = str(created_raw)

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{report_name}**")
                    st.caption(f"{dataset_size} samples · {created_str}")
                with col2:
                    if st.button("📄 View", key=f"view_{report_id}"):
                        st.session_state.selected_eval_report = report_id
                with col3:
                    if st.button("🗑️ Delete", key=f"del_{report_id}"):
                        if api_delete(f"/evaluate/reports/{report_id}"):
                            st.success(f"Deleted '{report_name}'")
                            st.rerun()
                        else:
                            st.error("Delete failed.")

                st.markdown("<hr style='border-color:#33334d;margin:8px 0;'>", unsafe_allow_html=True)

        # Show selected report details
        selected_id = st.session_state.get("selected_eval_report")
        if selected_id:
            report_detail = api_get(f"/evaluate/reports/{selected_id}")
            if report_detail:
                st.markdown("---")
                st.markdown(f"### 📄 {report_detail.get('report_name', 'Report')}")
                st.caption(f"Generated: {report_detail.get('created_at', '')}")

                summary = report_detail.get("summary", {})
                st.markdown("#### Aggregate Scores")
                sc1, sc2, sc3, sc4 = st.columns(4)
                for col, metric in zip([sc1, sc2, sc3, sc4], ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
                    val = summary.get(metric)
                    if val is not None:
                        col.metric(metric, f"{val:.4f}")
                st.metric("Total Samples", summary.get("rows", 0))

                samples = report_detail.get("samples", [])
                if samples:
                    st.markdown("#### Per-Sample Scores")
                    for i, sample in enumerate(samples, start=1):
                        with st.expander(f"Sample #{i}: {sample.get('question', '')[:80]}"):
                            st.markdown(f"**Question:** {sample.get('question', 'N/A')}")
                            st.markdown(f"**Answer:** {sample.get('answer', 'N/A')}")
                            st.markdown(f"**Ground Truth:** {sample.get('ground_truth', 'N/A')}")
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            for col, metric in zip([sc1, sc2, sc3, sc4], ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
                                val = sample.get(metric)
                                if val is not None:
                                    col.metric(metric, f"{val:.4f}")

                if st.button("Close Report"):
                    st.session_state.selected_eval_report = None
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

