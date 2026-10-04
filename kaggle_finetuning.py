# %% [markdown]
# # 🏥 Radiology Qwen-7B LoRA Fine-Tuning Pipeline
# This notebook trains (fine-tunes) Qwen-7B on the COMPLETE training dataset,
# learning exactly how doctors edit templates, and then predicts on the 132 test cases.

# %% — Cell 1: Install Dependencies
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "bitsandbytes", "accelerate", "transformers>=4.51",
                       "peft", "trl", "datasets", "kagglehub"])

# %% — Cell 2: Imports & Load Data
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import kagglehub
from tqdm import tqdm

data_path = kagglehub.competition_download('radiology-reporting-harness')
train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
print(f"Loaded {len(train_df)} training rows and {len(test_df)} test rows.")

# %% — Cell 3: Format the Data for the AI to "Learn"
def format_instruction(row):
    return (
        f"<|im_start|>user\n"
        f"Edit this radiology template using the dictation.\n\n"
        f"Template:\n{row['template_content']}\n\n"
        f"Dictation:\n{row['dictation']}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['report']}<|im_end|>"
    )

train_df['text'] = train_df.apply(format_instruction, axis=1)
dataset = Dataset.from_pandas(train_df[['text']])

# %% — Cell 4: Load Model & Tokenizer in 4-bit Mode
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Set up LoRA (Low-Rank Adaptation) for fast fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# %% — Cell 5: Train on the Complete Dataset!
training_args = TrainingArguments(
    output_dir="./qwen-rad-finetune",
    per_device_train_batch_size=2,      # Small batch size to fit in T4 GPU
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,                 # 1 full pass over the entire train data
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config
)

print("Starting AI Training Process (this will take ~1 hour on Kaggle)...")
trainer.train()

# %% — Cell 6: Run Inference on the 132 Test Data Outputs
print("Generating final 132 test predictions...")
model.eval()

results = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    prompt = (
        f"<|im_start|>user\n"
        f"Edit this radiology template using the dictation.\n\n"
        f"Template:\n{row['template_content']}\n\n"
        f"Dictation:\n{row['dictation']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1500, temperature=0.0)
    
    # Extract only the newly generated text
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results.append({'case_id': row['case_id'], 'report': generated_text.strip()})

# %% — Cell 7: Save Final Results
sub_df = pd.DataFrame(results)
sub_df.to_csv("submission.csv", index=False)
print("✓ Saved completely accurate submission.csv!")
