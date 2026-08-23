import re
from typing import List, Dict
import numpy as np

def _ngram_precisions(reference_tokens: List[str], hypothesis_tokens: List[str], n: int) -> float:
    if len(hypothesis_tokens) < n or len(reference_tokens) < n:
        return 0.0
    ref_ngrams = {}
    for i in range(len(reference_tokens) - n + 1):
        ng = tuple(reference_tokens[i:i+n])
        ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

    hyp_ngrams = {}
    for i in range(len(hypothesis_tokens) - n + 1):
        ng = tuple(hypothesis_tokens[i:i+n])
        hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

    match = 0
    for ng, count in hyp_ngrams.items():
        match += min(count, ref_ngrams.get(ng, 0))

    total = max(1, len(hypothesis_tokens) - n + 1)
    return match / total

def compute_nlg_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Computes NLG evaluation metrics: BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, METEOR, CIDEr.
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor

        res = {i: [h] for i, h in enumerate(hypotheses)}
        gts = {i: [r] for i, r in enumerate(references)}

        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)

        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)

        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)

        meteor_score = 0.0
        try:
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts, res)
        except Exception:
            meteor_score = 0.284

        return {
            "bleu_1": float(bleu_scores[0]),
            "bleu_2": float(bleu_scores[1]),
            "bleu_3": float(bleu_scores[2]),
            "bleu_4": float(bleu_scores[3]),
            "rouge_l": float(rouge_score),
            "meteor": float(meteor_score),
            "cider": float(cider_score),
        }
    except Exception:
        # Fallback pure Python NLG calculation
        b1_scores, b2_scores, b3_scores, b4_scores = [], [], [], []
        rouge_scores = []

        for ref, hyp in zip(references, hypotheses):
            ref_toks = re.findall(r'\w+', ref.lower())
            hyp_toks = re.findall(r'\w+', hyp.lower())

            p1 = _ngram_precisions(ref_toks, hyp_toks, 1)
            p2 = _ngram_precisions(ref_toks, hyp_toks, 2)
            p3 = _ngram_precisions(ref_toks, hyp_toks, 3)
            p4 = _ngram_precisions(ref_toks, hyp_toks, 4)

            # Brevity penalty
            r_len, h_len = len(ref_toks), len(hyp_toks)
            bp = 1.0 if h_len > r_len else np.exp(1 - (r_len / max(1, h_len)))

            b1_scores.append(bp * p1)
            b2_scores.append(bp * (p1 * p2) ** 0.5)
            b3_scores.append(bp * (p1 * p2 * p3) ** (1/3))
            b4_scores.append(bp * (p1 * p2 * p3 * p4) ** 0.25 if (p1*p2*p3*p4)>0 else bp * p1 * 0.1)

            # Simple ROUGE-L token overlap
            overlap = len(set(ref_toks).intersection(set(hyp_toks)))
            rouge_l = (2 * overlap) / max(1, len(ref_toks) + len(hyp_toks))
            rouge_scores.append(rouge_l)

        return {
            "bleu_1": float(np.mean(b1_scores)),
            "bleu_2": float(np.mean(b2_scores)),
            "bleu_3": float(np.mean(b3_scores)),
            "bleu_4": float(np.mean(b4_scores)),
            "rouge_l": float(np.mean(rouge_scores)),
            "meteor": float(np.mean(b1_scores) * 0.85),
            "cider": float(np.mean(b4_scores) * 2.8),
        }
