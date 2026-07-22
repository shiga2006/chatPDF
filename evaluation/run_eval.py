#!/usr/bin/env python3
"""
Convenience runner for RAGAS evaluation.

This script imports the evaluation logic and runs it with sensible defaults.
It's designed for quick iteration during development.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --judge-model gpt-4o --format html
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import from evaluation package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.ragas_eval import run_evaluation, main

if __name__ == "__main__":
    print("=" * 60)
    print("chatPDF - RAGAS Evaluation Runner")
    print("=" * 60)

    # Check for OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("\n⚠️  WARNING: OPENAI_API_KEY is not set.")
        print("   RAGAS evaluation requires an OpenAI API key to run the judge LLM.")
        print("   Set it with: set OPENAI_API_KEY=sk-... (Windows)")
        print("   Or: export OPENAI_API_KEY=sk-... (Mac/Linux)\n")
        proceed = input("   Continue anyway? (y/N): ").strip().lower()
        if proceed != "y":
            print("Exiting.")
            sys.exit(1)

    main()

