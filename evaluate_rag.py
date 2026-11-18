"""
RAG Pipeline Evaluation Script for SERC.
Calculates Precision@K, Recall@K, MRR (Mean Reciprocal Rank), and BERTScore (Precision, Recall, F1)
across gold-standard Wazuh security alert scenarios.
Outputs structured JSON and human-readable terminal table.

Usage:
    python backend/scripts/evaluate_rag.py
"""
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.embedding_service import embedding_service
from backend.services.retrieval_service import retrieval_service
from backend.services.pixtral_service import pixtral_service
from backend.services.gpt_analysis_service import gpt_analysis_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("serc.evaluate")


def compute_bert_score(candidate: str, reference: str) -> Tuple[float, float, float]:
    """
    Computes BERTScore (Precision, Recall, F1) between candidate and reference texts
    using contextual token embeddings from BGE/BERT transformer model with greedy cosine matching.
    """
    try:
        embedding_service.load_model()
        model = embedding_service.model
        if model is None or not hasattr(model, "tokenizer"):
            # Fallback to character/word n-gram cosine matching if model unavailable
            return _ngram_semantic_score(candidate, reference)

        import torch
        tokenizer = model.tokenizer
        transformer = model._first_module().auto_model

        # Tokenize with truncation to max_seq_length
        cand_tokens = tokenizer(candidate, return_tensors="pt", truncation=True, max_length=512)
        ref_tokens = tokenizer(reference, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            cand_out = transformer(**cand_tokens)
            ref_out = transformer(**ref_tokens)

        # Extract last hidden states: [1, seq_len, dim]
        cand_emb = cand_out.last_hidden_state.squeeze(0)  # [cand_len, dim]
        ref_emb = ref_out.last_hidden_state.squeeze(0)    # [ref_len, dim]

        # Exclude special tokens ([CLS], [SEP])
        if cand_emb.size(0) > 2:
            cand_emb = cand_emb[1:-1]
        if ref_emb.size(0) > 2:
            ref_emb = ref_emb[1:-1]

        # L2-normalize
        cand_norm = cand_emb / (torch.norm(cand_emb, dim=1, keepdim=True) + 1e-9)
        ref_norm = ref_emb / (torch.norm(ref_emb, dim=1, keepdim=True) + 1e-9)

        # Pairwise cosine similarity matrix: [cand_len, ref_len]
        sim_matrix = torch.matmul(cand_norm, ref_norm.T)

        # Greedy matching
        recall_val = torch.mean(torch.max(sim_matrix, dim=0)[0]).item()
        precision_val = torch.mean(torch.max(sim_matrix, dim=1)[0]).item()

        # Clip between 0 and 1
        precision_val = max(0.0, min(1.0, precision_val))
        recall_val = max(0.0, min(1.0, recall_val))

        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-9)
        return round(precision_val, 4), round(recall_val, 4), round(f1_val, 4)

    except Exception as e:
        logger.warning(f"Could not compute neural BERTScore ({e}), using semantic fallback: {e}")
        return _ngram_semantic_score(candidate, reference)


def _ngram_semantic_score(candidate: str, reference: str) -> Tuple[float, float, float]:
    """Fallback word-level semantic similarity calculator."""
    c_words = set(candidate.lower().split())
    r_words = set(reference.lower().split())
    if not c_words or not r_words:
        return 0.0, 0.0, 0.0

    overlap = len(c_words.intersection(r_words))
    p = overlap / len(c_words)
    r = overlap / len(r_words)
    f1 = 2 * p * r / (p + r + 1e-9)
    return round(p, 4), round(r, 4), round(f1, 4)


def is_chunk_relevant(chunk: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Evaluates whether a retrieved Qdrant chunk is genuinely relevant to the evaluation scenario.
    Checks MITRE technique ID, attack category, and domain keywords.
    """
    chunk_tid = str(chunk.get("mitre_technique_id", "")).strip()
    chunk_text = (chunk.get("chunk_text") or chunk.get("search_text") or "").lower()
    chunk_attack = str(chunk.get("attack_type", "")).lower()

    # 1. Direct MITRE Technique match
    for expected_tid in scenario.get("expected_mitre_techniques", []):
        if expected_tid.lower() in chunk_tid.lower() or expected_tid.lower() in chunk_text:
            return True, f"MITRE Technique Match ({expected_tid})"

    # 2. Attack type match
    expected_attack = scenario.get("attack_type", "").lower()
    if expected_attack and (expected_attack in chunk_attack or expected_attack in chunk_text):
        return True, f"Attack Type Match ({scenario.get('attack_type')})"

    # 3. Keyword matches
    kw_matches = [kw for kw in scenario.get("expected_keywords", []) if kw.lower() in chunk_text]
    if len(kw_matches) >= 2:
        return True, f"Domain Keywords Match ({', '.join(kw_matches[:2])})"

    return False, "Not relevant"


def run_evaluation() -> Dict[str, Any]:
    logger.info("=" * 65)
    logger.info("SERC RAG Retrieval & Response Evaluation Suite")
    logger.info("=" * 65)

    eval_file = PROJECT_ROOT / "backend/data/evaluation_dataset.json"
    if not eval_file.exists():
        logger.error(f"Evaluation dataset not found at {eval_file}")
        return {}

    with open(eval_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    # Load sample alerts
    samples_file = settings.sample_alerts_file
    with open(samples_file, "r", encoding="utf-8") as f:
        sample_alerts = {str(a.get("id")): a for a in json.load(f)}

    results = []
    p1_list, p3_list, p5_list = [], [], []
    r1_list, r3_list, r5_list = [], [], []
    mrr_list = []
    bert_p_list, bert_r_list, bert_f1_list = [], [], []

    for sc in scenarios:
        eval_id = sc["eval_id"]
        sc_name = sc["scenario_name"]
        alert_id = sc["alert_id"]
        expected_tids = sc.get("expected_mitre_techniques", [])
        ref_analysis = sc.get("reference_analysis", "")

        alert = sample_alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert {alert_id} not found in sample alerts, skipping.")
            continue

        # 1. Interpret alert
        pixtral_info = pixtral_service.interpret_alert(alert)
        search_query = pixtral_info.get("search_query") or alert.get("rule", {}).get("description", "")

        # 2. Retrieve top-5 evidence chunks from Qdrant
        retrieved_chunks = retrieval_service.search_evidence(query=search_query, top_k=5)

        # 3. Compute relevance flags for each retrieved chunk at rank 1..5
        relevance_flags = []
        relevance_reasons = []
        for ch in retrieved_chunks:
            rel, reason = is_chunk_relevant(ch, sc)
            relevance_flags.append(rel)
            relevance_reasons.append(reason)

        # Precision@K
        p1 = sum(relevance_flags[:1]) / 1.0 if retrieved_chunks else 0.0
        p3 = sum(relevance_flags[:3]) / 3.0 if len(retrieved_chunks) >= 3 else (sum(relevance_flags) / max(1, len(retrieved_chunks)))
        p5 = sum(relevance_flags[:5]) / 5.0 if len(retrieved_chunks) >= 5 else (sum(relevance_flags) / max(1, len(retrieved_chunks)))

        # Recall@K (Relative to minimum target benchmark of 3 relevant knowledge items)
        target_relevant_count = 3.0
        r1 = min(1.0, sum(relevance_flags[:1]) / 1.0)
        r3 = min(1.0, sum(relevance_flags[:3]) / target_relevant_count)
        r5 = min(1.0, sum(relevance_flags[:5]) / target_relevant_count)

        # MRR: Reciprocal rank of the first relevant chunk
        reciprocal_rank = 0.0
        for rank_idx, is_rel in enumerate(relevance_flags, start=1):
            if is_rel:
                reciprocal_rank = 1.0 / rank_idx
                break

        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)
        r1_list.append(r1)
        r3_list.append(r3)
        r5_list.append(r5)
        mrr_list.append(reciprocal_rank)

        # 4. Generate RAG Response
        generated_analysis = gpt_analysis_service.analyze_alert(
            alert=alert,
            pixtral_info=pixtral_info,
            evidence=retrieved_chunks
        )

        candidate_response = (
            f"{generated_analysis.get('alert_summary', '')} "
            f"Risk: {generated_analysis.get('risk_assessment', {}).get('level')}. "
            f"Reason: {generated_analysis.get('risk_assessment', {}).get('reason')}. "
            f"Observed: {' '.join(generated_analysis.get('observed_facts', []))}. "
            f"Interpretations: {' '.join(generated_analysis.get('possible_interpretations', []))}. "
            f"Investigation: {' '.join(generated_analysis.get('recommended_investigation_steps', []))}. "
            f"Response: {' '.join(generated_analysis.get('response_recommendations', []))}."
        )

        # 5. Compute BERTScore against expert reference answer
        bp, br, bf1 = compute_bert_score(candidate_response, ref_analysis)
        bert_p_list.append(bp)
        bert_r_list.append(br)
        bert_f1_list.append(bf1)

        results.append({
            "eval_id": eval_id,
            "scenario_name": sc_name,
            "alert_id": alert_id,
            "rule_id": alert.get("rule", {}).get("id"),
            "search_query": search_query,
            "retrieved_chunk_count": len(retrieved_chunks),
            "relevance_at_k": relevance_flags,
            "relevance_reasons": relevance_reasons,
            "precision_at_1": round(p1, 4),
            "precision_at_3": round(p3, 4),
            "precision_at_5": round(p5, 4),
            "recall_at_1": round(r1, 4),
            "recall_at_3": round(r3, 4),
            "recall_at_5": round(r5, 4),
            "reciprocal_rank": round(reciprocal_rank, 4),
            "bert_score": {
                "precision": bp,
                "recall": br,
                "f1": bf1
            }
        })

    # Aggregates
    summary = {
        "metrics_explanation": {
            "Precision@K": "Measures what proportion of the top-K retrieved knowledge chunks are relevant to the security alert.",
            "Recall@K": "Measures the fraction of relevant knowledge captured within the top-K retrieved results.",
            "MRR (Mean Reciprocal Rank)": "Measures the quality of ranking by averaging 1/rank of the first relevant result across all queries.",
            "BERTScore (P, R, F1)": "Measures semantic similarity between generated RAG analysis and gold-standard expert analyst responses using contextual embeddings."
        },
        "overall_results": {
            "mean_precision_at_1": round(float(np.mean(p1_list)), 4),
            "mean_precision_at_3": round(float(np.mean(p3_list)), 4),
            "mean_precision_at_5": round(float(np.mean(p5_list)), 4),
            "mean_recall_at_1": round(float(np.mean(r1_list)), 4),
            "mean_recall_at_3": round(float(np.mean(r3_list)), 4),
            "mean_recall_at_5": round(float(np.mean(r5_list)), 4),
            "mean_reciprocal_rank": round(float(np.mean(mrr_list)), 4),
            "mean_bert_score": {
                "precision": round(float(np.mean(bert_p_list)), 4),
                "recall": round(float(np.mean(bert_r_list)), 4),
                "f1": round(float(np.mean(bert_f1_list)), 4)
            }
        },
        "scenarios_evaluated": len(results),
        "detailed_results": results
    }

    # Save to disk
    out_file = PROJECT_ROOT / "backend/data/evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete. Results written to {out_file}")

    # Print Table
    print("\n" + "=" * 90)
    print("                    SERC RAG PIPELINE EVALUATION REPORT")
    print("=" * 90)
    print(f"{'Scenario Name':<32} | {'P@1':<5} | {'P@3':<5} | {'P@5':<5} | {'R@3':<5} | {'MRR':<5} | {'BERT-F1':<7}")
    print("-" * 90)
    for r in results:
        print(f"{r['scenario_name']:<32} | {r['precision_at_1']:<5.2f} | {r['precision_at_3']:<5.2f} | {r['precision_at_5']:<5.2f} | {r['recall_at_3']:<5.2f} | {r['reciprocal_rank']:<5.2f} | {r['bert_score']['f1']:<7.4f}")
    print("-" * 90)
    ov = summary["overall_results"]
    print(f"{'OVERALL AVERAGE':<32} | {ov['mean_precision_at_1']:<5.2f} | {ov['mean_precision_at_3']:<5.2f} | {ov['mean_precision_at_5']:<5.2f} | {ov['mean_recall_at_3']:<5.2f} | {ov['mean_reciprocal_rank']:<5.2f} | {ov['mean_bert_score']['f1']:<7.4f}")
    print("=" * 90)
    print("\nMetric Details:")
    print(f"  * Precision@K: P@1 = {ov['mean_precision_at_1']}, P@3 = {ov['mean_precision_at_3']}, P@5 = {ov['mean_precision_at_5']}")
    print(f"  * Recall@K:    R@1 = {ov['mean_recall_at_1']}, R@3 = {ov['mean_recall_at_3']}, R@5 = {ov['mean_recall_at_5']}")
    print(f"  * MRR:         {ov['mean_reciprocal_rank']}")
    print(f"  * BERTScore:   Precision={ov['mean_bert_score']['precision']}, Recall={ov['mean_bert_score']['recall']}, F1={ov['mean_bert_score']['f1']}\n")

    return summary


if __name__ == "__main__":
    run_evaluation()
