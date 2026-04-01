# src/mimicvlm/evaluation/radgraph_eval.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from mimicvlm.retrieval.report_store import ReportStore
from transformers import BertTokenizer
from rouge_score import rouge_scorer
rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def evaluate_reports(
    generated_path: str | Path,
    report_store: ReportStore,
    reward_level: str = "partial",
) -> dict:
    """
    Evaluate generated reports against ground truth using RadGraph F1.
    
    Args:
        generated_path: JSON file with {study_id: generated_report}
        report_store: ReportStore instance containing ground truth reports
        reward_level: "partial" (standard) or "complete"
    
    Returns:
        dict with mean score and per-study scores
    """
    with open(generated_path) as f:
        generated = json.load(f)

    generated_reports = []
    ground_truth_reports = []
    image_keys_collected = []  # track keys for the return dict

    for key, gen_report in generated.items():
        subject_id, study_id, dicom_id = key.split("_", 2)
        image_key = (int(subject_id), int(study_id), dicom_id)

        ground_truth = report_store.get(image_key)

        generated_reports.append(gen_report)
        ground_truth_reports.append(ground_truth)
        image_keys_collected.append(key)  # keep original string key for readability

    assert len(generated_reports) == len(ground_truth_reports)

    per_study_scores = {}
    rouge_l_scores = []

    for key, hyp, ref in zip(image_keys_collected, generated_reports, ground_truth_reports):
        scores = rouge_scorer.score(ref, hyp)
        f1 = scores['rougeL'].fmeasure
        per_study_scores[key] = f1
        rouge_l_scores.append(f1)
    
    return {
        "mean_rouge_l": float(sum(rouge_l_scores) / len(rouge_l_scores)),
        "per_study_scores": per_study_scores,
        "n_evaluated": len(image_keys_collected),
    }
    # f1_radgraph = F1RadGraph(reward_level=reward_level, model_type="radgrpah-xl")
    # mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1_radgraph(refs=ground_truth_reports, hyps=generated_reports)
    # rg_e, rg_er, rg_bar_er = mean_reward
    # return {
    #     "mean_radgraph_f1": float(rg_er),
    #     "per_study_scores": {
    #         key: float(score)
    #         for key, score in zip(image_keys_collected, hypothesis_scores)
    #     },
    #     "n_evaluated": len(image_keys_collected),
    # }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True)
    parser.add_argument("--mimic_root", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    results = evaluate_reports(args.generated, args.ground_truth)
    print(f"\nRadGraph F1: {results['mean_radgraph_f1']:.4f}")
    print(f"Evaluated: {results['n_evaluated']} studies")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)