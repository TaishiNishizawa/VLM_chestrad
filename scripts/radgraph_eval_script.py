# src/mimicvlm/evaluation/radgraph_eval.py
from __future__ import annotations
import json
import argparse
from pathlib import Path
from radgraph import F1RadGraph
import os

def evaluate_reports(
    generated_path: str | Path,
    mimic_root: str | Path,
    reward_level: str = "partial",
) -> dict:
    """
    Evaluate generated reports against ground truth using RadGraph F1.

    Args:
        generated_path: JSON file with {key: generated_report}
        mimic_root:     Root directory of MIMIC-CXR dataset
        reward_level:   "partial" (standard) or "complete"

    Returns:
        dict with mean score and per-study scores
    """
    mimic_root = Path(mimic_root)

    with open(generated_path) as f:
        generated = json.load(f)

    generated_reports = []
    ground_truth_reports = []
    keys_collected = []
    total = len(generated)

    for key, gen_report in generated.items():
        subject_id, study_id, dicom_id = key.split("_", 2)
        path = mimic_root / "files" / f"p{str(subject_id)[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"

        if not path.exists():
            print(f"Warning: ground truth not found for {key}, skipping.")
            continue

        ground_truth = path.read_text(encoding="utf-8", errors="replace").strip()

        generated_reports.append(gen_report)
        ground_truth_reports.append(ground_truth)
        keys_collected.append(key)


    assert len(generated_reports) == len(ground_truth_reports)
    print(f"Evaluating {len(generated_reports)} reports...")

    f1_radgraph = F1RadGraph(reward_level="partial", model_type="radgraph")
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1_radgraph(
        refs=ground_truth_reports, 
        hyps=generated_reports
    )

    return {
        "mean_radgraph_f1": float(mean_reward),
        "per_study_scores": {
            key: float(score)
            for key, score in zip(keys_collected, reward_list)
        },
        "n_evaluated": len(keys_collected),
        "n_skipped": len(generated) - len(keys_collected),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate radiology reports with RadGraph F1")
    parser.add_argument("--generated",    required=True, help="Path to generated reports JSON")
    parser.add_argument("--mimic_root",   required=True, help="Root directory of MIMIC-CXR dataset")
    parser.add_argument("--save_dir",     required=True, help="Directory to save results JSON")
    parser.add_argument("--reward_level", default="all", choices=["partial", "all"])
    args = parser.parse_args()

    results = evaluate_reports(args.generated, args.mimic_root, args.reward_level)

    print(f"\nRadGraph F1: {results['mean_radgraph_f1']:.4f}")
    print(f"Evaluated:   {results['n_evaluated']} studies")
    print(f"Skipped:     {results['n_skipped']} studies")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "evaluation_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")