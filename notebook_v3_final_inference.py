# ============================================================
# notebook_v3_final_inference.py
# 
# PURPOSE: 
# This is the final inference-ready version. It maintains 3 few-shot examples 
# but has the Auto-Prompt Optimizer DISABLED (AUTO_OPTIMIZE_PROMPT = False) 
# to save compute during the final run. The prompt rules are tailored to 
# ensure comprehensive coverage of all measurements without omitting any details.
# ============================================================

# ============================================================
# 🏥 Radiology Report Generation Pipeline
# Kaggle Competition: radiology-reporting-harness
# Model: Qwen3-14B (4-bit quantized) on Kaggle T4 GPU
# ============================================================
#
# 3-Stage Pipeline:
#   1. Few-Shot Retrieval — find similar train examples by modality + body_part
#   2. LLM Generation    — Qwen3-14B edits the template based on dictation
#   3. Post-Processing   — force-preserve unchanged template fields
#
# Structure: each "# %%" marks a new Kaggle notebook cell.
# Copy-paste into a Kaggle notebook, or upload as a script.
# ============================================================


# %% [markdown]
# # 🏥 Radiology Report Generation Pipeline
# **Model:** Qwen3-14B (4-bit quantized)  
# **Strategy:** Conservative template editing — only change what the dictation requires.


# %% — Cell 1: Install Dependencies
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "bitsandbytes", "accelerate", "transformers>=4.51",
                       "kagglehub"])


# %% — Cell 2: Imports & Configuration
# ============================================================
import os
import re
import gc
import time
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# ── Tunable Configuration ──────────────────────────────────
MODEL_ID           = "Qwen/Qwen2.5-7B-Instruct"      # Primary model (~5GB in 4-bit, fits T4 easily)
FALLBACK_MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"      # Fallback (even smaller, ~2.5GB)
MAX_NEW_TOKENS     = 2048      # Max output tokens per case
NUM_FEW_SHOT       = 3         # Increased Few-shot examples per case for more RAG context
ENABLE_THINKING    = False     # Qwen2.5 does not have thinking mode; set True only for Qwen3
SIMILARITY_THRESH  = 0.85      # Post-processing: force template if similarity > this
SAVE_EVERY         = 10        # Save intermediate results every N cases
VALIDATION_MODE    = False     # Set True to score against Train Data; False to run on Test Data
VAL_SAMPLE_SIZE    = 30        # Number of train cases to use for validation
AUTO_OPTIMIZE_PROMPT = False   # Automatically run the 5-loop evolutionary prompt optimizer!
OPTIMIZATION_SAMPLE_SIZE = 30  # Number of cases to learn from during prompt optimization
# ────────────────────────────────────────────────────────────

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB")


# %% — Cell 3: Download & Load Competition Data
# ============================================================
import kagglehub

data_path = kagglehub.competition_download('radiology-reporting-harness')
print(f"Data path: {data_path}")

train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_df  = pd.read_csv(os.path.join(data_path, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

print(f"\nTrain:  {train_df.shape[0]} rows × {train_df.shape[1]} cols")
print(f"Test:   {test_df.shape[0]} rows × {test_df.shape[1]} cols")
print(f"Sample: {sample_sub.shape[0]} rows × {sample_sub.shape[1]} cols")

print(f"\n--- Test Modality Distribution ---")
print(test_df['modality'].value_counts().to_string())
print(f"\n--- Test Body Part Distribution (top 10) ---")
print(test_df['body_part'].value_counts().head(10).to_string())


# %% — Cell 4: Few-Shot Retrieval System
# ============================================================
def build_retrieval_index(train_df):
    """Group training examples by (modality, body_part) for fast lookup."""
    exact_index = defaultdict(list)
    modality_index = defaultdict(list)

    for _, row in train_df.iterrows():
        exact_index[(row['modality'], row['body_part'])].append(row)
        modality_index[row['modality']].append(row)

    return exact_index, modality_index


def find_few_shot_examples(test_row, exact_index, modality_index, n=2):
    """Retrieve n training examples whose dictation is most similar to the test case."""
    test_dict = str(test_row.get('dictation', ''))
    key = (test_row['modality'], test_row['body_part'])
    
    # Try exact match first
    candidates = exact_index.get(key, [])
    # Fallback if not enough examples
    if len(candidates) < n:
        candidates.extend(modality_index.get(test_row['modality'], []))
    if len(candidates) < n:
        candidates = [r for rows in exact_index.values() for r in rows]

    # Prevent Data Leakage during validation: remove the test case itself from the candidates
    candidates = [c for c in candidates if c.get('case_id') != test_row.get('case_id')]

    if not candidates:
        return []

    # Calculate TF-IDF semantic similarity on the dictation text
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    corpus = [str(r.get('dictation', '')) for r in candidates]
    corpus.append(test_dict) # Add the test case to the end
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # Compare test dictation (last item) with all candidates
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
        # Get indices of the top N most similar training cases
        top_indices = similarities.argsort()[-n:][::-1]
        return [candidates[i] for i in top_indices]
    except ValueError:
        # Fallback if TF-IDF fails for some reason
        return candidates[:n]


# Build index once
if VALIDATION_MODE:
    # Strictly remove 50 cases from training data for validation
    val_df_true = train_df.sample(VAL_SAMPLE_SIZE, random_state=42).copy()
    train_df_retrieval = train_df.drop(val_df_true.index).copy()
else:
    train_df_retrieval = train_df

exact_idx, mod_idx = build_retrieval_index(train_df_retrieval)
print(f"Retrieval index ready: {len(exact_idx)} (modality, body_part) groups built from {len(train_df_retrieval)} training cases.")


# %% — Cell 5: Load Model
# ============================================================
def load_model(model_id):
    """Load a causal LM with BitsAndBytes NF4 4-bit quantization."""
    print(f"  Tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )

    # Kaggle T4 GPUs are Turing (Compute 7.5) — no flash_attention_2
    # PyTorch's native "sdpa" works well on T4
    attn = "sdpa"

    print(f"  Model:     {model_id} (BnB NF4 4-bit, attn={attn}) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn,
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  ✓ Loaded — VRAM used: {vram:.2f} GB")
    return model, tokenizer


print("Loading model …")
try:
    model, tokenizer = load_model(MODEL_ID)
    active_model = MODEL_ID
except Exception as e:
    print(f"  ✗ {MODEL_ID} failed: {e}")
    print(f"  → Trying fallback: {FALLBACK_MODEL_ID}")
    model, tokenizer = load_model(FALLBACK_MODEL_ID)
    active_model = FALLBACK_MODEL_ID
    ENABLE_THINKING = False  # Qwen2.5 does not support thinking mode

print(f"\nActive model: {active_model}")
print(f"Thinking mode: {ENABLE_THINKING}")


# %% — Cell 6: Prompt Engineering
# ============================================================
SYSTEM_PROMPT = """\
You are a radiology report template editor. Your ONLY task is to incorporate \
the doctor's dictated findings into the provided normal template to produce \
the final report.

CRITICAL RULES — follow every one exactly:

1. TEMPLATE IS YOUR STARTING REPORT.
   Copy every unchanged field using the EXACT same words and punctuation.
   Do NOT rephrase, reword, or "clean up" any text that does not need changing.

2. FIELD ROUTING.
   For each dictated finding, identify the ONE template field it belongs to \
based on anatomy. Edit ONLY that field. Never place a finding under the \
wrong anatomical label (e.g., pleural effusion → PLEURA, not LUNGS).

3. PARTIAL FIELD EDITING.
   When editing a field, preserve the parts of the normal statement that \
remain true.
   Example: Template says "No focal airspace opacity or pulmonary edema." \
Dictation mentions "right basilar opacity" but NOT edema.
   Correct: "Right basilar airspace opacity. No pulmonary edema."
   Wrong:   "Right basilar airspace opacity." (deleted the still-true edema \
statement)

4. NO HALLUCINATION.
   Never add findings not present in the dictation or template. \
Never infer diagnoses unless the dictation states them. \
modality, body_part, age, and sex are context only — they do NOT supply findings.

5. NEVER REMOVE TEMPLATE FIELDS.
   Every field label from the template must appear in your output, in the \
same order.

6. IMPRESSION.
   Concisely summarize ONLY the important abnormal findings from FINDINGS. \
Do not list normal findings. Do not introduce new information. \
If nothing is abnormal, keep the template's original impression text exactly.

7. OTHER FINDINGS.
   If the template has an "OTHER FINDINGS:" field, place relevant dictated \
findings there only if they don't belong in any other labelled field. \
Leave it empty when no such finding applies.

--- DYNAMIC RULES FROM AUTO-OPTIMIZER ---
1. ANATOMICAL ROUTING (BONES VS JOINTS): ALWAYS ensure findings related to joint conditions, alignment, and osteoarthrosis are under the "JOINTS" section. NEVER place them under "BONES" or "OSSEOUS STRUCTURES".
2. EXPLICIT NEGATIVES: ALWAYS explicitly state "No acute fracture" in the BONES section if there are no fractures. NEVER mix non-fracture bone findings directly under this statement; separate them clearly.
3. PREVENT HALLUCINATIONS: NEVER include sections or findings in the report that were not explicitly mentioned or implied by the given dictation.
4. IMPRESSION DISCIPLINE: NEVER move findings from the findings section into the "IMPRESSION" section. NEVER add "All findings are normal" to the IMPRESSION when the dictation is simply "normal".
5. NARRATIVE STRUCTURE: ALWAYS ensure that descriptive phrases like "There is" or "There are" precede findings to maintain proper narrative flow.
6. CHAIN OF THOUGHT: Before generating the final report, you MUST output a <plan> block. Inside <plan> ... </plan>, briefly list the findings from the dictation and which section they belong to.

7. ALWAYS include specific measurements and descriptions provided in the dictation within the findings section and repeat them accurately in the impression section if necessary.
8. ALWAYS include all findings for each anatomical region (vertebrae, disc spaces, soft tissues) before generating the impression.
9. ALWAYS ensure all findings for each anatomical region are fully described before generating the impression.
10. ALWAYS separate the findings of each anatomical region into distinct sections to maintain clarity and avoid mixing findings from different areas of the body.
11. NEVER omit any specific anatomical findings, ensuring comprehensive coverage of all relevant areas.

12. OUTPUT FORMAT.
   Return ONLY the report starting with "FINDINGS:" through the end of \
"IMPRESSION:". No markdown, no code blocks, no explanations, no preamble."""


def _build_user_message(row):
    """Format a single case into the user-prompt block."""
    return (
        f"[CONTEXT]\n"
        f"Modality: {row['modality']}\n"
        f"Body Part: {row['body_part']}\n"
        f"Study Description: {row['study_description']}\n"
        f"Patient Age: {row['patient_age_band']}\n"
        f"Patient Sex: {row['patient_sex']}\n"
        f"\n"
        f"[TEMPLATE TO EDIT]\n"
        f"{row['template_content']}\n"
        f"\n"
        f"[DOCTOR'S DICTATION]\n"
        f"{row['dictation']}\n"
        f"\n"
        f"Edit the template above by incorporating the dictation. "
        f"Copy unchanged fields exactly. Return FINDINGS and IMPRESSION only."
    )


def build_messages(test_row, few_shot_examples):
    """Assemble the full chat-message list: system → few-shot → test case."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Few-shot examples as prior user/assistant turns
    for ex in few_shot_examples:
        messages.append({"role": "user",      "content": _build_user_message(ex)})
        messages.append({"role": "assistant",  "content": str(ex['report'])})

    # Actual test case
    messages.append({"role": "user", "content": _build_user_message(test_row)})
    return messages


# %% — Cell 7: Report Parsing & Post-Processing Utilities
# ============================================================
_FIELD_RE = re.compile(r'^([A-Z][A-Z\s/&\-()]{1,60}?):\s*(.*)', re.MULTILINE)


def parse_report_fields(text):
    """Parse a radiology report into an ordered list of (label, content).

    Handles multi-line fields. Skips the bare "FINDINGS:" section header
    (which has sub-fields) but keeps "IMPRESSION:" as a field.
    """
    results = []
    current_label = None
    current_content = []

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        m = _FIELD_RE.match(stripped)
        if m:
            # Flush previous field
            if current_label and current_label != "FINDINGS":
                results.append((current_label, ' '.join(current_content).strip()))

            current_label = m.group(1).strip()
            rest = m.group(2).strip()
            current_content = [rest] if rest else []
        elif current_label and current_label != "FINDINGS":
            current_content.append(stripped)

    # Flush last field
    if current_label and current_label != "FINDINGS":
        results.append((current_label, ' '.join(current_content).strip()))

    return results


def _word_set(text):
    return set(text.lower().split())


def _jaccard(text1, text2):
    """Jaccard word-set similarity."""
    s1, s2 = _word_set(text1), _word_set(text2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def strip_thinking(text):
    """Remove <think>…</think> blocks from Qwen3 thinking-mode output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_report_text(raw):
    """Pull FINDINGS…IMPRESSION from raw model output."""
    raw = strip_thinking(raw)

    # Try exact case first
    idx = raw.find("FINDINGS:")
    if idx == -1:
        # Case-insensitive fallback: find position then slice from there
        upper = raw.upper()
        idx = upper.find("FINDINGS:")
    if idx == -1:
        return raw.strip()

    report = raw[idx:].strip()
    # Ensure FINDINGS: is uppercase in the output
    if not report.startswith("FINDINGS:"):
        report = "FINDINGS:" + report[len("FINDINGS:"):]
    return report


def post_process(generated_report, template_text):
    """Enforce template fidelity on the generated report.

    For every field whose generated text is very close to the template
    (Jaccard > SIMILARITY_THRESH), replace with EXACT template text.
    This eliminates the most common LLM failure: subtle rephrasing
    of normal fields.
    """
    tmpl_fields = parse_report_fields(template_text)
    gen_fields  = parse_report_fields(generated_report)

    if not tmpl_fields or not gen_fields:
        return generated_report  # fallback: can't parse

    gen_lookup = {label: content for label, content in gen_fields}

    # Separate IMPRESSION
    tmpl_findings  = [(l, c) for l, c in tmpl_fields if l != "IMPRESSION"]
    tmpl_impression = next((c for l, c in tmpl_fields if l == "IMPRESSION"), "")

    gen_impression  = gen_lookup.get("IMPRESSION", "")

    # ── Reconstruct FINDINGS ────────────────────────────────
    lines = ["FINDINGS:"]

    for label, tmpl_content in tmpl_findings:
        gen_content = gen_lookup.get(label, tmpl_content)

        sim = _jaccard(gen_content, tmpl_content)
        if sim > SIMILARITY_THRESH:
            # LLM barely changed this → use exact template wording
            lines.append(f"{label}: {tmpl_content}")
        else:
            # LLM made meaningful edits → keep its version
            lines.append(f"{label}: {gen_content}")

    # Handle fields the LLM added that aren't in template
    tmpl_labels = {l for l, _ in tmpl_findings}
    for label, content in gen_fields:
        if label not in tmpl_labels and label not in ("IMPRESSION", "FINDINGS"):
            if content and content.strip():
                lines.append(f"{label}: {content}")

    # ── Reconstruct IMPRESSION ──────────────────────────────
    lines.append("")
    lines.append("IMPRESSION:")

    if tmpl_impression and gen_impression:
        sim = _jaccard(gen_impression, tmpl_impression)
        if sim > SIMILARITY_THRESH:
            lines.append(tmpl_impression)
        else:
            lines.append(gen_impression)
    elif gen_impression:
        lines.append(gen_impression)
    else:
        lines.append(tmpl_impression)

    return '\n'.join(lines)


# %% — Cell 8: Single-Case Generation
# ============================================================
def generate_single(row, model, tokenizer, few_shot_examples):
    """Generate one report: prompt → LLM → extract → post-process."""

    messages = build_messages(row, few_shot_examples)

    # ── Tokenize ────────────────────────────────────────────
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # model.device can fail with device_map="auto" on multi-GPU
    # Use the device of the first parameter instead
    target_device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(target_device)
    input_len = inputs["input_ids"].shape[1]

    # ── Generate ────────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
        )

    new_ids = output_ids[0][input_len:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # ── Extract & post-process ──────────────────────────────
    report = extract_report_text(raw_text)
    report = post_process(report, str(row['template_content']))

    # ── Self-Correction Critic Pass ─────────────────────────
    critic_prompt = (
        f"<|im_start|>system\nYou are a senior radiology auditor.<|im_end|>\n"
        f"<|im_start|>user\nReview this generated report against the original dictation.\n"
        f"DICTATION: {row['dictation']}\n\n"
        f"GENERATED REPORT:\n{report}\n\n"
        f"Did the AI miss any findings from the dictation or misplace them in the wrong section? "
        f"If yes, output the CORRECTED report starting with 'FINDINGS:'. "
        f"If the report is perfectly accurate, output the original report exactly as is starting with 'FINDINGS:'.\n"
        f"Output ONLY the final report, no explanations.<|im_end|>\n"
        f"<|im_start|>assistant\nFINDINGS:"
    )
    
    critic_inputs = tokenizer(critic_prompt, return_tensors="pt").to(target_device)
    with torch.no_grad():
        critic_out = model.generate(
            **critic_inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=False,
            repetition_penalty=1.0
        )
        
    final_report = "FINDINGS:" + tokenizer.decode(critic_out[0][critic_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Free memory
    del inputs, output_ids
    torch.cuda.empty_cache()

    return final_report


# %% — Cell 9: Full Inference Pipeline
# ============================================================
def run_pipeline(test_df, train_df, model, tokenizer):
    """Iterate through every test case and produce reports."""
    results = []
    t0 = time.time()

    for i, (idx, row) in enumerate(
        tqdm(test_df.iterrows(), total=len(test_df), desc="Generating")
    ):
        try:
            # Stage 1 — Retrieve similar examples
            examples = find_few_shot_examples(
                row, exact_idx, mod_idx, n=NUM_FEW_SHOT
            )

            # Stage 2 + 3 — Generate & post-process
            report = generate_single(row, model, tokenizer, examples)

            results.append({"case_id": row["case_id"], "report": report})

        except Exception as e:
            print(f"\n⚠  Error on {row['case_id'][:12]}…: {e}")
            # Safe fallback: return the unedited template
            results.append({
                "case_id": row["case_id"],
                "report": str(row["template_content"]),
            })

        # ── Progress & intermediate save ────────────────────
        if (i + 1) % SAVE_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test_df) - i - 1) / rate
            print(
                f"\n  [{i+1}/{len(test_df)}]  "
                f"Elapsed {elapsed/60:.1f} min · ETA {eta/60:.1f} min"
            )
            pd.DataFrame(results).to_csv(
                "intermediate_submission.csv", index=False
            )

    elapsed = time.time() - t0
    print(f"\n✓ Done — {len(results)} reports in {elapsed/60:.1f} min")
    return pd.DataFrame(results)

def score_report(truth, pred):
    global model, tokenizer
    try:
        import torch, re
        target_device = next(model.parameters()).device
        prompt = (
            "<|im_start|>system\nYou are an expert medical evaluator.<|im_end|>\n"
            "<|im_start|>user\nRate the clinical similarity of the PREDICTED report compared to the TRUTH report. "
            "Ignore formatting and exact wording. Focus on whether the medical findings and diagnoses match. "
            "Provide ONLY a single float number between 0.0 and 1.0 (e.g. 0.95) and absolutely nothing else.\n\n"
            f"TRUTH:\n{truth}\n\n"
            f"PREDICTED:\n{pred}<|im_end|>\n"
            "<|im_start|>assistant\nScore: "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                repetition_penalty=1.0,
            )
        
        new_ids = output_ids[0][input_len:]
        raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        
        match = re.search(r"(?:0\.\d+|1\.0+)", raw_text)
        if match:
            return float(match.group())
        elif raw_text.strip() == "1":
            return 1.0
        elif raw_text.strip() == "0":
            return 0.0
        else:
            raise ValueError(f"Could not parse score from: {raw_text}")
    except Exception as e:
        import difflib
        return difflib.SequenceMatcher(None, str(truth), str(pred)).ratio()


# %% — Cell 9.5: Auto-Prompt Optimization Phase
# ============================================================
if AUTO_OPTIMIZE_PROMPT:
    print("=" * 60)
    print(f"  PHASE 1: EVOLUTIONARY AUTO-PROMPT (5 Loops on {OPTIMIZATION_SAMPLE_SIZE} Cases)")
    print("=" * 60)
    
    # --- HARD NEGATIVE MINING ---
    print("Mining for the hardest medical cases...")
    def calc_difficulty(row):
        t_set = set(str(row['template_content']).lower().split())
        r_set = set(str(row['report']).lower().split())
        if not t_set or not r_set: return 1.0
        return len(t_set & r_set) / len(t_set | r_set)
        
    train_df_retrieval['difficulty_score'] = train_df_retrieval.apply(calc_difficulty, axis=1)
    
    # 1. Sample the absolute hardest cases from the training pool
    opt_df = train_df_retrieval.sort_values('difficulty_score').head(OPTIMIZATION_SAMPLE_SIZE).copy()
    
    # We will validate on the hidden validation set
    val_opt_df = val_df_true.head(VAL_SAMPLE_SIZE).copy()
    
    best_score = 0.0
    best_prompt = SYSTEM_PROMPT
    
    target_device = next(model.parameters()).device
    
    for loop_idx in range(5):
        print(f"\n{'*'*40}")
        print(f"   AUTO-OPTIMIZE LOOP {loop_idx + 1}/5")
        print(f"{'*'*40}")
        
        # 1. Run pipeline on the hard cases
        print("  -> Generating reports to find mistakes...")
        opt_results = run_pipeline(opt_df, train_df_retrieval, model, tokenizer)
        opt_merged = opt_df.merge(opt_results, on='case_id', suffixes=('_truth', '_pred'))
        
        generated_rules = []
        for _, row in opt_merged.iterrows():
            acc = score_report(row['report_truth'], row['report_pred'])
            if acc < 0.95:
                analyzer_prompt = (
                    f"<|im_start|>system\nYou are an expert AI Prompt Engineer.<|im_end|>\n"
                    f"<|im_start|>user\nOur AI made a mistake editing a radiology template.\n\n"
                    f"TEMPLATE GIVEN:\n{row['template_content']}\n\n"
                    f"DICTATION GIVEN:\n{row['dictation']}\n\n"
                    f"WHAT THE AI GENERATED (MISTAKE):\n{row['report_pred']}\n\n"
                    f"WHAT IT WAS SUPPOSED TO GENERATE (CORRECT):\n{row['report_truth']}\n\n"
                    f"TASK: Identify exactly what the AI did wrong. "
                    f"Based on this mistake, write ONE strict, one-sentence rule (starting with 'ALWAYS' or 'NEVER') that we can add to the AI's system prompt to prevent this exact mistake in the future.<|im_end|>\n"
                    f"<|im_start|>assistant\nRule:"
                )
                
                inputs_a = tokenizer(analyzer_prompt, return_tensors="pt").to(target_device)
                with torch.no_grad():
                    out_rule = model.generate(**inputs_a, max_new_tokens=150, do_sample=False)
                rule = tokenizer.decode(out_rule[0][inputs_a.input_ids.shape[1]:], skip_special_tokens=True).strip()
                generated_rules.append(rule)
                
        if generated_rules:
            unique_rules = list(set(generated_rules))
            print(f"  -> Generated {len(unique_rules)} raw rules! Condensing to optimum 5...")
            
            all_rules_text = "\n".join([f"- {r}" for r in unique_rules])
            distill_prompt = (
                f"<|im_start|>system\nYou are an expert Prompt Engineer.<|im_end|>\n"
                f"<|im_start|>user\nHere are raw rules our AI generated to fix its mistakes:\n{all_rules_text}\n\n"
                f"Some of these rules might be repetitive or contradictory. "
                f"Synthesize them into a final list of exactly the 5 most critical, distinct, and strict rules. "
                f"Output ONLY the numbered list, nothing else.<|im_end|>\n"
                f"<|im_start|>assistant\n1."
            )
            
            inputs_d = tokenizer(distill_prompt, return_tensors="pt").to(target_device)
            with torch.no_grad():
                out_distill = model.generate(**inputs_d, max_new_tokens=300, do_sample=False)
            distilled_rules = "1. " + tokenizer.decode(out_distill[0][inputs_d.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            print(f"\n[+] Top 5 Condensed Rules:\n{distilled_rules}\n")
            
            # Inject the rules into SYSTEM_PROMPT (replacing old dynamic rules if present)
            import re
            if "--- DYNAMIC RULES FROM AUTO-OPTIMIZER ---" in SYSTEM_PROMPT:
                SYSTEM_PROMPT = re.sub(
                    r"--- DYNAMIC RULES FROM AUTO-OPTIMIZER ---.*?8\. OUTPUT FORMAT\.",
                    f"--- DYNAMIC RULES FROM AUTO-OPTIMIZER ---\n{distilled_rules}\n\n8. OUTPUT FORMAT.",
                    SYSTEM_PROMPT,
                    flags=re.DOTALL
                )
            else:
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
                    "8. OUTPUT FORMAT.",
                    f"--- DYNAMIC RULES FROM AUTO-OPTIMIZER ---\n{distilled_rules}\n\n8. OUTPUT FORMAT."
                )
                
            # Now Validate the new SYSTEM_PROMPT on the validation set!
            print("  -> Testing the new 5 prompt rules on validation set...")
            val_results = run_pipeline(val_opt_df, train_df_retrieval, model, tokenizer)
            val_merged = val_opt_df.merge(val_results, on='case_id', suffixes=('_truth', '_pred'))
            val_merged['score'] = val_merged.apply(lambda r: score_report(r['report_truth'], r['report_pred']), axis=1)
            current_score = val_merged['score'].mean()
            
            print(f"  => Validation Accuracy for Loop {loop_idx + 1}: {current_score:.4f}")
            
            if current_score > best_score:
                print("  [✓] Accuracy improved! Saving as new best prompt.")
                best_score = current_score
                best_prompt = SYSTEM_PROMPT
            else:
                print(f"  [✗] Accuracy did not improve (best was {best_score:.4f}). Reverting to best prompt for next loop.")
                SYSTEM_PROMPT = best_prompt
                
        else:
            print("\n[+] No mistakes found in optimization phase. Prompt is perfect!")
            break
            
    print("=" * 60)
    print(f"EVOLUTIONARY LOOP FINISHED. Best Validation Accuracy: {best_score:.4f}")
    print("=" * 60)
    SYSTEM_PROMPT = best_prompt


# %% — Cell 10: Validation on Train Data
# ============================================================
if VALIDATION_MODE:
    print("=" * 60)
    print(f"  VALIDATION MODE (Evaluating on {VAL_SAMPLE_SIZE} hidden cases)")
    print("=" * 60)
    
    # Run the pipeline on the 50 removed validation cases
    val_results_df = run_pipeline(val_df_true, train_df_retrieval, model, tokenizer)
    
    # Score the results against the ground truth
    print("\n--- Calculating Validation Score ---")
    val_scored = val_df_true.merge(val_results_df, on='case_id', suffixes=('_truth', '_pred'))
    
    val_scored['score'] = val_scored.apply(lambda r: score_report(r['report_truth'], r['report_pred']), axis=1)
    
    print(f"\n🏆 VALIDATION SCORE (Accuracy similarity): {val_scored['score'].mean():.4f}")
    print("Note: 1.0 is perfect. A higher score means it perfectly matched the doctor's actual report.")
    
    print("\n--- Worst Performing Case (To see what went wrong) ---")
    worst = val_scored.loc[val_scored['score'].idxmin()]
    print(f"Modality/Body Part: {worst['modality']} - {worst['body_part']}")
    print(f"Score: {worst['score']:.4f}")
    print("\nDICTATION:\n", worst['dictation'])
    print("\nMODEL GENERATED:\n", str(worst['report_pred']).split('IMPRESSION:')[0][:300], "...")
    print("\nGROUND TRUTH:\n", str(worst['report_truth']).split('IMPRESSION:')[0][:300], "...")


# %% — Cell 11: Run Full Inference on Test Data (Real Submission)
# ============================================================
if not VALIDATION_MODE:
    print("=" * 60)
    print(f"  TEST MODE (Generating for {len(test_df)} test cases)")
    print(f"  Model:           {active_model}")
    print(f"  Few-shot (n):    {NUM_FEW_SHOT}")
    print(f"  Post-proc thresh:{SIMILARITY_THRESH}")
    print("=" * 60)

    submission_df = run_pipeline(test_df, train_df, model, tokenizer)


# %% — Cell 12: Validate & Save Submission (Real Submission Only)
# ============================================================
if not VALIDATION_MODE:
    # ── Sanity checks ───────────────────────────────────────────
    assert len(submission_df) == len(test_df), \
        f"Row count mismatch: {len(submission_df)} vs {len(test_df)}"
    assert set(submission_df['case_id']) == set(test_df['case_id']), \
        "case_id mismatch between submission and test"

    missing_findings = 0
    missing_impression = 0
    for _, row in submission_df.iterrows():
        r = str(row['report']).upper()
        if 'FINDINGS:' not in r:
            missing_findings += 1
            print(f"⚠  Missing FINDINGS: {row['case_id'][:12]}…")
        if 'IMPRESSION:' not in r:
            missing_impression += 1
            print(f"⚠  Missing IMPRESSION: {row['case_id'][:12]}…")

    print(f"\nValidation results:")
    print(f"  Total reports:      {len(submission_df)}")
    print(f"  Missing FINDINGS:   {missing_findings}")
    print(f"  Missing IMPRESSION: {missing_impression}")

    # ── Save final CSV ──────────────────────────────────────────
    submission_df[['case_id', 'report']].to_csv("submission.csv", index=False)
    print(f"\n✓ Saved submission.csv ({len(submission_df)} rows)")

    # ── Preview ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PREVIEW — First 3 reports:")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(submission_df.head(3).iterrows()):
        print(f"\n--- Case {row['case_id'][:12]}… ---")
        print(row['report'][:400])
        if len(row['report']) > 400:
            print("…")
        print()
