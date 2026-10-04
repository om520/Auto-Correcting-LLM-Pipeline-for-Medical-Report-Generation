# Radiology Report Validation Metric: Jaccard Similarity

In this pipeline, we measure the performance of our LLM-generated radiology reports against the ground-truth doctor's reports using the **Jaccard Similarity** index.

## The Equation

The mathematical formula for Jaccard Similarity (Intersection over Union) is:

$$J(A, B) = \frac{| A \cap B |}{| A \cup B |}$$

In plain English, when applied to text:
**Score = (Number of words that appear in BOTH reports) ÷ (Total unique words across BOTH reports combined)**

### Python Implementation
```python
def score_report(truth, pred):
    s1, s2 = set(str(truth).lower().split()), set(str(pred).lower().split())
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)
```

## Why use Jaccard Similarity?

The Kaggle competition relies on an **Edit Distance** metric (such as Levenshtein distance) to heavily penalize models that rewrite normal template fields. However, computing the exact Levenshtein edit distance for thousands of words across many cases in a Jupyter notebook is computationally slow.

Jaccard Similarity serves as an incredibly fast and accurate **proxy** for edit distance:
* **Score of `1.0` (100%)**: The LLM produced the exact same words as the real doctor (perfect score).
* **Score of `~0.8` (80%)**: The LLM captured most of the right words but hallucinated some or missed a few.
* **Score of `< 0.3` (< 30%)**: The LLM completely rewrote the report and failed to preserve the template's structure.

By tracking this Jaccard score on a validation set (e.g., 50 random samples from `train.csv`), you can confidently predict whether a prompt tweak or post-processing rule will raise or lower your final Kaggle leaderboard rank.
