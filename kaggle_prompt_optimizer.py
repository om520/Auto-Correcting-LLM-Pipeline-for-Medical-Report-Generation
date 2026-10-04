# %% [markdown]
# # 🧠 Radiology Auto-Prompt Optimizer (Realistic RAG Version)
# This script uses the 14B model to test itself on 30 hidden cases, catch its own mistakes,
# and automatically generate new strict rules to add to your System Prompt!

# %% — Cell 1: Install Dependencies
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "bitsandbytes", "accelerate", "transformers>=4.51",
                       "kagglehub"])

# %% — Cell 2: Imports & Configuration
import os
import torch
import pandas as pd
import difflib
import kagglehub
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
SAMPLE_SIZE = 30  # Test on 30 cases (Hidden from the training data)
NUM_FEW_SHOT = 2  # Same as main notebook

print("Loading Data...")
data_path = kagglehub.competition_download('radiology-reporting-harness')
train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))

# Create 30 Hidden Test Cases (Remove them from the training data!)
hidden_test_df = train_df.sample(SAMPLE_SIZE, random_state=42).copy()
clean_train_df = train_df.drop(hidden_test_df.index).copy()

# %% — Cell 3: Build the TF-IDF Retrieval Index
def build_retrieval_index(df):
    exact_index = defaultdict(list)
    modality_index = defaultdict(list)
    for _, row in df.iterrows():
        exact_index[(row['modality'], row['body_part'])].append(row)
        modality_index[row['modality']].append(row)
    return exact_index, modality_index

exact_idx, mod_idx = build_retrieval_index(clean_train_df)

def find_few_shot_examples(test_row):
    test_dict = str(test_row.get('dictation', ''))
    key = (test_row['modality'], test_row['body_part'])
    
    candidates = exact_idx.get(key, [])
    if len(candidates) < NUM_FEW_SHOT:
        candidates.extend(mod_idx.get(test_row['modality'], []))
    if len(candidates) < NUM_FEW_SHOT:
        candidates = [r for rows in exact_idx.values() for r in rows]

    if not candidates:
        return []

    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = [str(r.get('dictation', '')) for r in candidates]
    corpus.append(test_dict)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        top_indices = similarities.argsort()[-NUM_FEW_SHOT:][::-1]
        return [candidates[i] for i in top_indices]
    except ValueError:
        return candidates[:NUM_FEW_SHOT]

# %% — Cell 4: Load Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
model.eval()

# %% — Cell 5: The Core Generation & Optimization Loop
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

generated_rules = []

print(f"\nStarting Self-Correction Loop on {SAMPLE_SIZE} Hidden Cases...\n")

for idx, row in tqdm(hidden_test_df.iterrows(), total=len(hidden_test_df)):
    # Retrieve RAG examples
    examples = find_few_shot_examples(row)
    
    # 1. Build Prompt (Mirroring the main notebook)
    prompt = f"<|im_start|>system\nYou are a strict radiology template editor.<|im_end|>\n"
    for ex in examples:
        prompt += (f"<|im_start|>user\nEdit this template using the dictation.\nTEMPLATE:\n{ex['template_content']}\nDICTATION:\n{ex['dictation']}<|im_end|>\n"
                   f"<|im_start|>assistant\n{ex['report']}<|im_end|>\n")
    
    prompt += (f"<|im_start|>user\nEdit this template using the dictation.\nTEMPLATE:\n{row['template_content']}\nDICTATION:\n{row['dictation']}<|im_end|>\n"
               f"<|im_start|>assistant\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    
    generated_report = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    truth_report = str(row['report']).strip()
    
    # 2. Check accuracy
    accuracy = score_report(truth_report, generated_report)
    
    # 3. If it made a mistake, force it to analyze its failure and write a rule!
    if accuracy < 0.95:
        analyzer_prompt = (
            f"<|im_start|>system\nYou are an expert AI Prompt Engineer.<|im_end|>\n"
            f"<|im_start|>user\nOur AI made a mistake editing a radiology template.\n\n"
            f"TEMPLATE GIVEN:\n{row['template_content']}\n\n"
            f"DICTATION GIVEN:\n{row['dictation']}\n\n"
            f"WHAT THE AI GENERATED (MISTAKE):\n{generated_report}\n\n"
            f"WHAT IT WAS SUPPOSED TO GENERATE (CORRECT):\n{truth_report}\n\n"
            f"TASK: Identify exactly what the AI did wrong. Did it put a finding in the wrong section? Did it delete something it shouldn't have? "
            f"Based on this mistake, write ONE strict, one-sentence rule (starting with 'ALWAYS' or 'NEVER') that we can add to the AI's system prompt to prevent this exact mistake in the future.<|im_end|>\n"
            f"<|im_start|>assistant\nRule:"
        )
        
        inputs_analyzer = tokenizer(analyzer_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_rule = model.generate(**inputs_analyzer, max_new_tokens=150, do_sample=False)
            
        rule = tokenizer.decode(out_rule[0][inputs_analyzer.input_ids.shape[1]:], skip_special_tokens=True).strip()
        generated_rules.append(rule)
        
        print(f"\n[Mistake Caught! Accuracy: {accuracy:.2f}]")
        print(f"Generated Rule: {rule}")

# %% — Cell 6: Print Final Optimized Rules
print("\n" + "="*60)
print("🏆 FINAL OPTIMIZED PROMPT RULES GENERATED BY THE AI:")
print("="*60)
unique_rules = list(set(generated_rules))
for i, rule in enumerate(unique_rules):
    print(f"{i+1}. {rule}")
print("\n(Copy and paste the best rules into the SYSTEM_PROMPT of your kaggle_notebook.py file!)")
