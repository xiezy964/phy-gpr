import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src import data_prep, step4_ablations
from src.utils import RANDOM_SEED, ensure_output_dirs, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mandatory ablations (spec-05).")
    parser.add_argument(
        "--test-groups",
        type=int,
        default=6,
        help="Number of original groups to hold out for test split (default: 5).",
    )
    parser.add_argument(
        "--baseline-restarts",
        type=int,
        default=10,
        help="Restart count for tuned baseline GP in Step4 ablations.",
    )
    parser.add_argument(
        "--physics-restarts",
        type=int,
        default=10,
        help="Restart count for tuned physics-informed GP in Step4 ablations.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    set_seed(RANDOM_SEED)
    data = data_prep.prepare(test_group_count=args.test_groups)
    print(
        f"=== Step4 Ablations === train={len(data['train'])} test={len(data['test'])} "
        f"test_groups={data['test_groups']}"
    )
    df = step4_ablations.run(
        data,
        seed=RANDOM_SEED,
        tuned_baseline_restarts=args.baseline_restarts,
        tuned_physics_restarts=args.physics_restarts,
    )
    print(f"wrote outputs/step4_ablation_metrics.csv with {len(df)} rows")


if __name__ == "__main__":
    main()
