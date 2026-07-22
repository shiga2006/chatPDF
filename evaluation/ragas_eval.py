#!/usr/bin/env python3
"""
RAGAS evaluation script for chatPDF agent outputs.

Usage:
    # Basic usage with default sample dataset
    python evaluation/ragas_eval.py

    # With custom input and output paths
    python evaluation/ragas_eval.py --input evaluation/custom_dataset.jsonl --output evaluation/reports/my_report.json

    # Export formats: json (default), markdown, html
    python evaluation/ragas_eval.py --format markdown --verbose

    # Use different judge / embedding models
    python evaluation/ragas_eval.py --judge-model gpt-4o --embedding-model text-embedding-3-large
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_records(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        return records

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    raise ValueError("Input must be a JSON array or a JSONL file.")


def _normalize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    required_fields = ["question", "answer", "contexts", "ground_truth"]
    for idx, item in enumerate(records, start=1):
        missing = [key for key in required_fields if key not in item]
        if missing:
            raise ValueError(
                f"Record {idx} is missing required fields: {', '.join(missing)}"
            )

        contexts = item.get("contexts")
        if not isinstance(contexts, list) or not all(isinstance(x, str) for x in contexts):
            raise ValueError(f"Record {idx} field 'contexts' must be a list of strings.")

        normalized.append(
            {
                "question": str(item.get("question", "")).strip(),
                "answer": str(item.get("answer", "")).strip(),
                "contexts": contexts,
                "ground_truth": str(item.get("ground_truth", "")).strip(),
            }
        )

    if not normalized:
        raise ValueError("No valid records found in the evaluation input.")

    return normalized


def _resolve_metrics() -> List[Any]:
    from ragas import metrics as ragas_metrics

    metric_name_options = [
        ["faithfulness"],
        ["answer_relevancy", "answer_relevance"],
        ["context_precision"],
        ["context_recall"],
    ]

    resolved: List[Any] = []
    missing_groups: List[str] = []

    for options in metric_name_options:
        metric_obj = None
        for name in options:
            metric_obj = getattr(ragas_metrics, name, None)
            if metric_obj is not None:
                resolved.append(metric_obj)
                break
        if metric_obj is None:
            missing_groups.append("/".join(options))

    if missing_groups:
        raise RuntimeError(
            "Unable to resolve these RAGAS metrics in your installed version: "
            + ", ".join(missing_groups)
        )

    return resolved


def _build_wrapped_llm_and_embeddings(
    judge_model: str, embedding_model: str
):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(model=judge_model, temperature=0.0)
    embeddings = OpenAIEmbeddings(model=embedding_model)

    llm_wrapper = None
    emb_wrapper = None

    try:
        from ragas.llms import LangchainLLMWrapper

        llm_wrapper = LangchainLLMWrapper(llm)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize RAGAS LangChain LLM wrapper. "
            "Check your ragas version and dependencies."
        ) from exc

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        emb_wrapper = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        try:
            from ragas.embeddings.base import LangchainEmbeddingsWrapper

            emb_wrapper = LangchainEmbeddingsWrapper(embeddings)
        except Exception as second_exc:
            raise RuntimeError(
                "Could not initialize RAGAS LangChain embeddings wrapper. "
                "Check your ragas version and dependencies."
            ) from second_exc

    return llm_wrapper, emb_wrapper


def _sanitize(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _format_markdown_report(payload: Dict[str, Any]) -> str:
    """Format the evaluation results as a Markdown report."""
    lines: List[str] = []
    lines.append("# RAGAS Evaluation Report")
    lines.append("")
    lines.append(f"- **Generated:** {payload.get('generated_at', 'N/A')}")
    lines.append(f"- **Input File:** {payload.get('input_file', 'N/A')}")
    lines.append(f"- **Judge Model:** {payload.get('judge_model', 'N/A')}")
    lines.append(f"- **Embedding Model:** {payload.get('embedding_model', 'N/A')}")
    lines.append("")

    summary = payload.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Average Score |")
    lines.append(f"|--------|--------------:|")
    for metric_name in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        value = summary.get(metric_name)
        if value is not None:
            if metric_name == "answer_relevancy" and metric_name not in summary:
                value = summary.get("answer_relevance")
            lines.append(f"| {metric_name} | {value:.4f} |")
    lines.append("")
    lines.append(f"**Total samples:** {summary.get('rows', 0)}")
    lines.append("")

    samples = payload.get("samples", [])
    if samples:
        lines.append("## Per-Sample Scores")
        lines.append("")
        for i, sample in enumerate(samples, start=1):
            lines.append(f"### Sample #{i}")
            lines.append("")
            lines.append(f"- **Question:** {sample.get('question', 'N/A')}")
            lines.append(f"- **Answer:** {sample.get('answer', 'N/A')}")
            lines.append(f"- **Ground Truth:** {sample.get('ground_truth', 'N/A')}")
            lines.append("")
            lines.append("| Metric | Score |")
            lines.append("|--------|------:|")
            for metric_name in [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]:
                value = sample.get(metric_name)
                if value is not None:
                    lines.append(f"| {metric_name} | {value:.4f} |")
            lines.append("")

    return "\n".join(lines)


def _format_html_report(payload: Dict[str, Any]) -> str:
    """Format the evaluation results as a self-contained HTML report."""
    summary = payload.get("summary", {})

    metric_rows = ""
    for metric_name in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        value = summary.get(metric_name)
        if value is not None:
            pct = value * 100
            color = "#22c55e" if value >= 0.7 else "#eab308" if value >= 0.4 else "#ef4444"
            metric_rows += f"""
            <tr>
                <td style="padding: 8px 16px; border-bottom: 1px solid #e5e7eb; font-weight: 500;">{metric_name}</td>
                <td style="padding: 8px 16px; border-bottom: 1px solid #e5e7eb;">
                    <div style="background: #e5e7eb; border-radius: 999px; height: 20px; width: 200px; overflow: hidden;">
                        <div style="background: {color}; height: 100%; width: {pct}%; border-radius: 999px;"></div>
                    </div>
                </td>
                <td style="padding: 8px 16px; border-bottom: 1px solid #e5e7eb; text-align: right; font-family: monospace; color: {color};">{value:.4f}</td>
            </tr>"""

    samples_html = ""
    for i, sample in enumerate(payload.get("samples", []), start=1):
        sample_metrics = ""
        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            value = sample.get(metric_name)
            if value is not None:
                sample_metrics += f"<tr><td style='padding:4px 12px;border-bottom:1px solid #f3f4f6;'>{metric_name}</td><td style='padding:4px 12px;border-bottom:1px solid #f3f4f6;text-align:right;font-family:monospace;'>{value:.4f}</td></tr>"

        samples_html += f"""
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:16px;">
            <h4 style="margin:0 0 8px 0;color:#111827;">Sample #{i}</h4>
            <p style="margin:4px 0;font-size:13px;"><strong>Q:</strong> {sample.get('question','')}</p>
            <p style="margin:4px 0;font-size:13px;"><strong>A:</strong> {sample.get('answer','')}</p>
            <p style="margin:4px 0;font-size:13px;"><strong>GT:</strong> {sample.get('ground_truth','')}</p>
            <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:13px;">{sample_metrics}</table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGAS Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f3f4f6; margin: 0; padding: 20px; color: #1f2937; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 24px; margin-bottom: 20px; }}
        h1 {{ margin: 0 0 4px 0; font-size: 24px; }}
        .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 16px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ text-align: left; padding: 8px 16px; border-bottom: 2px solid #e5e7eb; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #6b7280; }}
        .score-badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>RAGAS Evaluation Report</h1>
            <div class="meta">
                <div>Generated: {payload.get('generated_at', 'N/A')}</div>
                <div>Judge: {payload.get('judge_model', 'N/A')} | Embedding: {payload.get('embedding_model', 'N/A')}</div>
                <div>Samples: {summary.get('rows', 0)}</div>
            </div>
            <h3>Aggregate Scores</h3>
            <table>
                <thead><tr><th>Metric</th><th>Score</th><th style="text-align:right;">Value</th></tr></thead>
                <tbody>{metric_rows}</tbody>
            </table>
        </div>
        <div class="card">
            <h3>Per-Sample Breakdown</h3>
            {samples_html}
        </div>
    </div>
</body>
</html>"""


def run_evaluation(
    input_path: Path,
    output_path: Path,
    judge_model: str,
    embedding_model: str,
    export_format: str = "json",
    verbose: bool = False,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for RAGAS evaluation in this script."
        )

    records = _normalize_records(_read_records(input_path))
    print(f"Loaded {len(records)} evaluation records from {input_path}")

    from datasets import Dataset

    dataset = Dataset.from_list(records)
    metrics = _resolve_metrics()
    print(f"Resolved metrics: {[m.name if hasattr(m, 'name') else str(m) for m in metrics]}")

    llm_wrapper, emb_wrapper = _build_wrapped_llm_and_embeddings(
        judge_model, embedding_model
    )

    from ragas import evaluate

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=emb_wrapper,
        raise_exceptions=False,
    )

    try:
        frame = result.to_pandas()
        per_sample = frame.to_dict(orient="records")
    except Exception:
        per_sample = getattr(result, "scores", [])

    if not isinstance(per_sample, list):
        per_sample = []

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "answer_relevance",
        "context_precision",
        "context_recall",
    ]

    summary: Dict[str, Any] = {
        "rows": len(per_sample),
    }

    for metric_name in metric_names:
        values = []
        for row in per_sample:
            value = row.get(metric_name)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                values.append(float(value))
        if values:
            summary[metric_name] = round(sum(values) / len(values), 4)

    payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_file": str(input_path),
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "summary": summary,
        "samples": _sanitize(per_sample),
    }

    # Write output based on format
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON report saved: {output_path}")

    elif export_format == "markdown":
        md_path = output_path.with_suffix(".md")
        with md_path.open("w", encoding="utf-8") as f:
            f.write(_format_markdown_report(payload))
        print(f"Markdown report saved: {md_path}")

    elif export_format == "html":
        html_path = output_path.with_suffix(".html")
        with html_path.open("w", encoding="utf-8") as f:
            f.write(_format_html_report(payload))
        print(f"HTML report saved: {html_path}")

    # Verbose per-sample breakdown
    if verbose:
        print("\n" + "=" * 60)
        print("PER-SAMPLE BREAKDOWN")
        print("=" * 60)
        for i, sample in enumerate(per_sample, start=1):
            print(f"\n--- Sample #{i} ---")
            print(f"  Question   : {sample.get('question', 'N/A')[:80]}...")
            print(f"  Faithfulness    : {sample.get('faithfulness', 'N/A')}")
            print(f"  Answer Relevancy: {sample.get('answer_relevancy', sample.get('answer_relevance', 'N/A'))}")
            print(f"  Context Precision: {sample.get('context_precision', 'N/A')}")
            print(f"  Context Recall  : {sample.get('context_recall', 'N/A')}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for chatPDF agent outputs."
    )
    parser.add_argument(
        "--input",
        default="evaluation/sample_eval_dataset.jsonl",
        help="Path to JSONL/JSON file with question, answer, contexts, ground_truth",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/ragas_report.json",
        help="Path for report output",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI model used as evaluation judge",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model used by RAGAS",
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "markdown", "html"],
        help="Output format for the report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample score breakdown to stdout",
    )
    args = parser.parse_args()

    output = run_evaluation(
        input_path=Path(args.input),
        output_path=Path(args.output),
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        export_format=args.format,
        verbose=args.verbose,
    )

    print("\nRAGAS evaluation completed.")
    print("Summary:")
    print(json.dumps(output.get("summary", {}), indent=2))
    print(f"\nReport saved.")


if __name__ == "__main__":
    main()

