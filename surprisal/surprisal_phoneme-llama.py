from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
from transformers import AutoTokenizer, AutoModelForCausalLM

import math
import pandas as pd
import torch
import torch.nn.functional as F

# Model Card

model_name = "bbunzeck/phoneme-llama"

# Load Model and Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Function to Compute Surprisal for a Specific Word

def compute_surprisal_for_word(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors = "pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"][0]

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    target_tokens = tokenizer.tokenize(target_word)
    target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_token_ids_tensor = torch.tensor(target_token_ids).to(device)

    for i in range(1, len(input_ids) - len(target_token_ids) + 1):
        if torch.equal(input_ids[i:i+len(target_token_ids)], target_token_ids_tensor):
            surprisal = 0
            for j, token_id in enumerate(target_token_ids):
                log_probs = F.log_softmax(logits[i - 1 + j], dim = -1)
                surprisal += -log_probs[token_id].item() / math.log(2)
            return round(surprisal, 4)
    return None

# Function to Compute Surprisal of Last Word in a Sentence

def compute_last_word_surprisal(sentence):
    words = sentence.strip().split()
    if len(words) >= 1:
        last_word = words[-1].strip(".,!?")
        return compute_surprisal_for_word(sentence, last_word)
    return None

# Sample Sentences to Test

sentences = [
    "Who do you wanna take to the station tomorrow?",
    "Who do you wanna go to the station tomorrow?"
]

# Run Tests

print("Word-by-word surprisal for each sample sentence:\n")

for sent in sentences:
    print(f"Sentence: {sent}")
    words = sent.strip().split()
    for word in words:
        clean_word = word.strip(".,!?")
        surprisal = compute_surprisal_for_word(sent, clean_word)
        print(f"  {clean_word}: {surprisal}")
    print()

# Load Excel File with 'SENTENCE' Column

df = pd.read_excel(".xlsx")

# Compute Surprisal for Last Word in Each Sentence

df["LAST_WORD_SURPRISAL"] = df["SENTENCE"].apply(compute_last_word_surprisal)

# Save Results

output_file = ".xlsx"
df.to_excel(output_file, index = False)
