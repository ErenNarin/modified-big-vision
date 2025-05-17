from bert_score import score
import language_tool_python
import textstat
import gensim.downloader as api
from big_vision.custom_evaluation import gensim_model
from big_vision.custom_evaluation.utils import compute_hybrid_score
from rouge_score import rouge_scorer

'''
| ----------- | ------------------------------------- |
| Metric      | Purpose                               |
| ----------- | ------------------------------------- |
| BERTScore   | Semantic similarity (deep context)    |
| ROGUE       | Longest N-gram overlap                |
| Grammar     | Fluency and correctness               |
| Readability | Clarity and ease of understanding     |
| ----------- | ------------------------------------- |
  MoverScore              https://www.galileo.ai/blog/moverscore-ai-semantic-text-evaluation
  Word Mover's Distance   https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html#word-mover-s-distance
'''


def evaluation_score(reference, candidate):
    wm_distance = calc_wmd_score(reference, candidate)
    bert_p, bert_r, bert_f1 = calc_bert_score(reference, candidate)
    rogue_scores = calc_rogue_score(reference, candidate)
    grammar_score = calc_grammar_score(candidate)
    readability_score = calc_readability_score(candidate)

    '''
    print("=== Evaluation Results ===")
    print(f"World Mover's Distance Score: {normalize('wmd', wm_distance):.2f}")
    print(f"BERT Score: {normalize('bertscore', F1):.2f}")
    print(f"Rogue Score: {normalize('rogue', rogue_scores):.2f}")
    print(f"Grammar Score: {normalize('grammar', grammar_score):.2f}")
    print(f"Flesch Reading Ease Score: {normalize('readability', readability_score):.2f}")
    '''

    metrics = {
        'wmd': wm_distance,
        'bertscore': bert_f1,
        'rogue': rogue_scores,
        'grammar': grammar_score,
        'readability': readability_score
    }

    weights = {
        'wmd': 0.35,
        'bertscore': 0.35,
        'rogue': 0.20,
        'grammar': 0.05,
        'readability': 0.05
    }

    hybrid_score = compute_hybrid_score(metrics, weights)
    # print(f"Evaluation Score: {hybrid_score:.4f}")

    return hybrid_score


def calc_wmd_score(reference, candidate):
    if 'gensim_model' not in globals() or gensim_model is None:
        gensim_model = api.load('word2vec-google-news-300')
        print("Gensim model loaded.")
    wm_distance = gensim_model.wmdistance(candidate, reference)
    print("Word Mover’s Distance: ", wm_distance)


def calc_bert_score(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return P, R, F1


def calc_rogue_score(reference, candidate):
    rogue_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rogue_scores = rogue_scorer.score(candidate, reference)
    return rogue_scores


def calc_grammar_score(candidate):
    tool = language_tool_python.LanguageTool('en-US')
    grammar_errors = tool.check(candidate)
    grammar_score = len(grammar_errors)
    return grammar_score


def calc_readability_score(candidate):
    readability_score = textstat.flesch_reading_ease(candidate)
    return readability_score
