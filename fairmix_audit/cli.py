from __future__ import annotations

import argparse
from pathlib import Path

from fairmix_audit.experiments import run_experiment


def run() -> None:
    parser = argparse.ArgumentParser(description="Run the fairness audit workflow.")
    parser.add_argument("--config", default="configs/smoke.yml", help="Path to a YAML config.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="How many split plans to batch together per work item.",
    )
    parser.add_argument("--output-root", help="Optional output root override.")
    args = parser.parse_args()
    run_dir = run_experiment(args.config, chunk_size=args.chunk_size, output_root=args.output_root)
    print(f"Wrote run artifacts to {run_dir}")


def tables() -> None:
    parser = argparse.ArgumentParser(description="Create compact tables from a run directory.")
    parser.add_argument("run_dir", help="Directory produced by the audit workflow.")
    args = parser.parse_args()
    from fairmix_audit.reporting import write_analysis_artifacts, write_model_cards, write_plots, write_tables

    run_dir = Path(args.run_dir)
    write_analysis_artifacts(run_dir)
    write_tables(run_dir)
    write_plots(run_dir)
    write_model_cards(run_dir)

    print(f"Wrote reporting artifacts to {run_dir}")


if __name__ == "__main__":
    run()
