# RAGAS Evaluation Report

- **Generated:** 2024-01-15T10:30:00Z
- **Input File:** evaluation/sample_eval_dataset.jsonl
- **Judge Model:** gpt-4o-mini
- **Embedding Model:** text-embedding-3-small

## Summary

| Metric | Average Score |
|--------|--------------:|
| faithfulness | 0.9245 |
| answer_relevancy | 0.8912 |
| context_precision | 0.8734 |
| context_recall | 0.8567 |

**Total samples:** 14

## Per-Sample Scores

### Sample #1

- **Question:** What is the annual carry-forward limit for unused leave?
- **Answer:** Employees can carry forward up to 10 leave days.
- **Ground Truth:** Up to 10 leave days can be carried forward.

| Metric | Score |
|--------|------:|
| faithfulness | 0.9542 |
| answer_relevancy | 0.8921 |
| context_precision | 0.9103 |
| context_recall | 0.8750 |

### Sample #2

- **Question:** Who should employees contact for payroll disputes?
- **Answer:** Payroll Operations at payroll@company.com.
- **Ground Truth:** Contact Payroll Operations.

| Metric | Score |
|--------|------:|
| faithfulness | 0.9831 |
| answer_relevancy | 0.9543 |
| context_precision | 0.9321 |
| context_recall | 0.9102 |
