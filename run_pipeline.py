import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src import data_prep, step1_gpr, step2_gpr, step3_optim
from src.report import write_technical_report
from src.utils import RANDOM_SEED, ensure_output_dirs, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physics-informed GPR pipeline.")
    parser.add_argument(
        "--test-groups",
        type=int,
        default=6,
        help="Number of original groups to hold out for test split (default: 5).",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    set_seed(RANDOM_SEED)

    print("=== Data preparation ===")
    data = data_prep.prepare(test_group_count=args.test_groups)
    print(
        f"  train={len(data['train'])} test={len(data['test'])} "
        f"train_groups={len(data['train_groups'])} test_groups={len(data['test_groups'])} "
        f"(requested test_groups={args.test_groups})"
    )

    print("\n=== Step 1: Dual GPR [P,v] -> [d, phi] ===")
    s1_models, s1_results = step1_gpr.run(data, optimize=True, n_restarts_optimizer=1)

    print("\n=== Step 2: Physics-informed GPR [P,v] -> UTS ===")
    s2_models, s2_results = step2_gpr.run(data, s1_models)

    print("\n=== Step 3: Inverse optimization ===")
    s3_df = step3_optim.run(s2_models, s1_models, data)

    write_technical_report(data, s1_results, s2_results, s3_df)
    print("\nAll outputs written to outputs/")


if __name__ == "__main__":
    main()
