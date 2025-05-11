from bert_score import score
import language_tool_python
import textstat
from big_vision.custom_evaluation.utils import format_for_coco, compute_hybrid_score
from pycocoevalcap.bleu.bleu import Bleu
from rouge_score import rouge_scorer


def evaluation_score(reference, candidate):
    bert_p, bert_r, bert_f1 = calc_bert_score(reference, candidate)
    grammar_score = calc_grammar_score(candidate)
    readability_score = calc_readability_score(candidate)
    bleu_score = calc_bleu_score(reference, candidate)
    rogue_score = calc_rogue_score(reference, candidate)

    '''
    print("=== Evaluation Results ===")
    print(f"BERTScore F1: {bert_f1[0]:.4f}")
    print(f"BLEU-1 to 4: {[round(s, 4) for s in bleu_score]}")
    print(f"Rogue Scores: {[s.fmeasure for m, s in rogue_score.items()]}")
    print(f"Grammar Issues: {grammar_score}")
    print(f"Flesch Reading Ease: {readability_score:.2f}")
    '''

    metrics = {
        'bertscore': bert_f1[0],
        'bleu': bleu_score,
        'rogue': rogue_score,
        'grammar': grammar_score,
        'readability': readability_score
    }

    weights = {
        'bertscore': 0.40,
        'bleu': 0.20,
        'rogue': 0.20,
        'grammar': 0.10,
        'readability': 0.10
    }

    hybrid_score = compute_hybrid_score(metrics)
    #print(f"Evaluation Score: {hybrid_score:.4f}")

    return hybrid_score


def calc_bert_score(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return P, R, F1


def calc_grammar_score(candidate):
    tool = language_tool_python.LanguageTool('en-US')
    grammar_errors = tool.check(candidate)
    grammar_score = len(grammar_errors)
    return grammar_score


def calc_readability_score(candidate):
    readability_score = textstat.flesch_reading_ease(candidate)
    return readability_score


def calc_bleu_score(reference, candidate):
    reference_list = {
        "0": [reference]
    }
    candidate_list = {
        "0": [candidate]
    }
    # COCO-compatible format
    gts = format_for_coco(reference_list)
    res = format_for_coco(candidate_list)

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    return bleu_scores


def calc_rogue_score(reference, candidate):
    rogue_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rogue_scores = rogue_scorer.score(candidate, reference)
    return rogue_scores