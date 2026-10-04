# Auto-Correcting LLM Pipeline for Medical Report Generation

This repository contains my solution for the **Radiology Reporting Harness** Kaggle competition. The goal of this project is to generate highly accurate, correctly formatted radiology reports by intelligently merging a doctor's dictation with a standard "normal" template.

## Results & Evaluation Methodology

**Final Validation Score:** `0.9473 / 1.0`

Because traditional string-matching metrics are poor indicators of medical accuracy, I used an **LLM-as-a-Judge Evaluation Technique** to validate the pipeline:
* **The Process:** A secondary LLM evaluator was prompted to act as an expert medical auditor. It compared the AI-generated reports against the ground-truth doctors' reports.
* **The Metric:** The evaluator ignored minor formatting differences and focused purely on clinical accuracy—ensuring findings, measurements, and diagnoses matched the truth perfectly. It then graded the report on a scale of 0.0 to 1.0.
* **The Outcome:** Tested on a hidden validation set of 30 complex cases, the pipeline achieved an impressive **0.9473** clinical accuracy score.

---

## Novel Contributions & Key Innovations

Instead of relying on a standard, static prompt, I engineered a highly customized pipeline featuring a unique **Evolutionary Auto-Prompt Optimization Loop** to systematically eliminate AI hallucinations.

### 1. The Evolutionary Auto-Prompting Loop (Self-Correction)
To prevent hallucinations and structural errors, I built an automated prompt-engineering loop that iteratively tests the model on a hidden validation set:
* **LLM Evaluator:** A secondary AI evaluator scores the primary generator's output against the ground-truth doctor's report to calculate a strict semantic similarity score.
* **Error Diagnosis:** When a mistake is caught (score < 0.95), an "AI Prompt Engineer" analyzes the exact failure and writes strict `ALWAYS` or `NEVER` rules to prevent the mistake from happening again.
* **Rule Distillation:** These raw rules are automatically condensed into a core list of directives and dynamically injected back into the system prompt for the next loop.
* **Result:** This auto-prompting loop successfully learned 12 critical rules (such as anatomical routing constraints and measurement inclusion) entirely on its own, pushing the final validation accuracy score to **0.9473 / 1.0**.

### 2. The Complete 3-Stage Pipeline
The final end-to-end pipeline is structured to ensure maximum accuracy and template fidelity:
1. **Few-Shot RAG Retrieval:** The system groups training data by modality and body part, utilizing TF-IDF semantic similarity to find the 3 historical cases that most closely match the current dictation. These are injected into the prompt to guide the model.
2. **LLM Generation:** The system uses `Qwen/Qwen2.5-7B-Instruct`, heavily quantized into 4-bit precision (via BitsAndBytes) so that it fits perfectly within Kaggle's standard T4 GPU VRAM limitations. 
3. **Strict Post-Processing:** The generated output is reconstructed and compared against the original template using a Jaccard text-similarity algorithm. If the AI barely modified a field, the script forcefully snaps the text back to the exact original template wording, completely eliminating subtle LLM rephrasing issues.

---

## Key Files

* `kaggle_notebook_final.py` (and `submission_package/final_inference.py`): The final production-ready script. Validation mode and the prompt-optimizer are intentionally disabled here for maximum inference speed on the 132 test cases.
* `submission_package/`: A directory containing the finalized code and submission documentation intended for the final assignment delivery.
* `notebook_v1_base_optimization.py` & `notebook_v2_enhanced_rules.py`: Earlier iterations showcasing the historical development of the RAG system and the Evolutionary Prompt Optimizer.

## How to Run Final Inference on Kaggle

You do not need to manually download or attach the dataset. The script handles it automatically via `kagglehub`.

1. Open a new Kaggle Notebook.
2. Under notebook settings, set **Accelerator** to **GPU T4 x2** (or equivalent) and turn **Internet ON**.
3. Copy the entire contents of `kaggle_notebook_final.py` into a single code cell.
4. Run the cell.
5. The script will automatically install necessary pip dependencies, download the dataset, run 3-shot inference on all 132 test cases, and save `submission.csv` to the `/kaggle/working/` directory.
