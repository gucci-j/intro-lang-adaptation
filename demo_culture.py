#!/usr/bin/env python3
"""
Working lecture demo for Section 5 (Culture):
Demonstrates two cultural limitations of LLMs on Qwen3.5-9B (non-thinking mode):
  1. Defaulting to dominant (often US) norms when asked in English without country cues
     (e.g. Mother's Day in May vs UK's March, emergency numbers, tax year, vocabulary).
  2. Weak everyday cultural knowledge of underrepresented cultures — especially when
     queried in the local language (Ethiopian culture in Amharic vs English;
     Welsh culture in Welsh vs English).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


US_DEFAULT_PROMPTS = [
    {
        "id": "mothers_day",
        "topic": "Mother's Day date",
        "prompt": "When is Mother's Day celebrated? Answer in one short sentence.",
        "us_norm": "Second Sunday of May",
        "uk_norm": "Mothering Sunday (fourth Sunday of Lent, in March)",
    },
    {
        "id": "emergency_number",
        "topic": "Emergency phone number",
        "prompt": "What number should I dial in an emergency? Answer in one short sentence.",
        "us_norm": "911",
        "uk_norm": "999 (or 112)",
    },
    {
        "id": "tax_year",
        "topic": "Tax year start",
        "prompt": "When does the new tax year start? Answer in one short sentence.",
        "us_norm": "January 1",
        "uk_norm": "April 6",
    },
    {
        "id": "pants_meaning",
        "topic": "Everyday vocabulary ('pants')",
        "prompt": "What piece of clothing does the word 'pants' refer to? Answer in one short sentence.",
        "us_norm": "Trousers",
        "uk_norm": "Underwear",
    },
    {
        "id": "first_floor",
        "topic": "Building floor numbering",
        "prompt": "If I walk into a building from the street, which floor am I on? Answer in one short sentence.",
        "us_norm": "First floor",
        "uk_norm": "Ground floor",
    },
]

ETHIOPIAN_CULTURE_PROMPTS = [
    {
        "id": "ethiopian_months",
        "topic": "Ethiopian calendar months",
        "gold_fact": "13 months (12 months of 30 days + Pagume of 5 or 6 days).",
        "en": "How many months are in the Ethiopian calendar, and what is the last month called? Answer briefly.",
        "am": "በኢትዮጵያ የዘመን አቆጣጠር ስንት ወራት አሉ? የመጨረሻው ወርስ ምን ይባላል? በአጭሩ መልስ።",
    },
    {
        "id": "enkutatash",
        "topic": "Enkutatash (Ethiopian New Year)",
        "gold_fact": "Meskerem 1 (September 11, or September 12 in a leap year); children sing 'Abebayehosh' and give Adey Abeba flowers.",
        "en": "What is Enkutatash, when is it celebrated, and what flower is traditionally associated with it? Answer briefly.",
        "am": "እንቁጣጣሽ ምንድን ነው? መቼ ይከበራል? ከበዓሉ ጋር የሚያያዘው ባህላዊ አበባስ ምን ይባላል? በአጭሩ መልስ።",
    },
    {
        "id": "buna_rounds",
        "topic": "Ethiopian coffee ceremony (Buna) rounds",
        "gold_fact": "Three rounds: Abol (first), Tona (second), and Baraka (third).",
        "en": "In a traditional Ethiopian coffee ceremony (Buna), what are the names of the three rounds of coffee? Answer briefly.",
        "am": "በባህላዊው የኢትዮጵያ የቡና ሥነ ሥርዓት ውስጥ የሦስቱ የቡና ዙሮች ስም ምን ምን ናቸው? በአጭሩ መልስ።",
    },
    {
        "id": "iddir_equb",
        "topic": "Iddir and Equb (social institutions)",
        "gold_fact": "Iddir is a community mutual-aid association mainly for funerals/bereavement; Equb is a rotating savings and credit association (ROSCA).",
        "en": "What is the difference between Iddir and Equb in Ethiopian everyday life? Answer briefly.",
        "am": "በኢትዮጵያ ማኅበረሰብ ውስጥ በዕድር እና በዕቁብ መካከል ያለው ልዩነት ምንድን ነው? በአጭሩ መልስ።",
    },
    {
        "id": "meskel_damera",
        "topic": "Meskel & Damera",
        "gold_fact": "Meskel (Meskerem 17 / Sept 27) commemorates the finding of the True Cross; people light a large bonfire called Damera.",
        "en": "What is the Damera in the Ethiopian festival of Meskel? Answer briefly.",
        "am": "በኢትዮጵያ የመስቀል በዓል ላይ 'ደመራ' ምንድን ነው? በአጭሩ መልስ።",
    },
]

WELSH_CULTURE_PROMPTS = [
    {
        "id": "santes_dwynwen",
        "topic": "Dydd Santes Dwynwen",
        "gold_fact": "25 January — Welsh equivalent of Valentine's Day, celebrating the Welsh patron saint of lovers.",
        "en": "When is Dydd Santes Dwynwen celebrated and what does it mark? Answer briefly.",
        "cy": "Pryd mae Dydd Santes Dwynwen yn cael ei ddathlu a beth mae'n ei nodi? Atebwch yn fyr.",
    },
    {
        "id": "mari_lwyd",
        "topic": "Mari Lwyd",
        "gold_fact": "A Welsh midwinter/New Year folk custom involving a decorated horse's skull carried door-to-door with rhyming song battles (pwnco).",
        "en": "What is the Mari Lwyd tradition in Wales? Answer briefly.",
        "cy": "Beth yw traddodiad y Fari Lwyd yng Nghymru? Atebwch yn fyr.",
    },
    {
        "id": "calennig",
        "topic": "Calennig",
        "gold_fact": "A Welsh New Year's Day custom where children go door-to-door singing for small gifts/money, traditionally carrying a decorated apple on three sticks.",
        "en": "What is the Welsh tradition of Calennig? Answer briefly.",
        "cy": "Beth yw'r traddodiad Cymreig o Galennig? Atebwch yn fyr.",
    },
]


def load_model_and_tokenizer(model_id: str):
    print(f"[culture] Loading {model_id}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", trust_remote_code=True
        )
    except Exception:
        if AutoModelForImageTextToText is None:
            raise
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map="auto", trust_remote_code=True
        )
    model.eval()
    print(f"[culture] Loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def generate_one(model, tokenizer, user_text: str, max_new_tokens: int = 200) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    in_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def translate_to_english(model, tokenizer, text: str, src_lang_name: str) -> str:
    prompt = (
        f"Translate the following {src_lang_name} text faithfully into English, "
        f"preserving any factual errors or hallucinations as written. Output only the translation.\n\n{text}"
    )
    return generate_one(model, tokenizer, prompt, max_new_tokens=220)


def main():
    hf_token_path = Path("/content/.hf_token")
    if hf_token_path.exists() and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token_path.read_text().strip()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", default="/content/culture_results.json")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    # --- Part 1: US-default norms ---
    print("\n=== PART 1: DEFAULT CULTURAL NORMS (ENGLISH, NO COUNTRY SPECIFIED) ===", flush=True)
    us_default_out = []
    for item in US_DEFAULT_PROMPTS:
        ans = generate_one(model, tokenizer, item["prompt"], max_new_tokens=120)
        us_default_out.append({**item, "response": ans})
        print(f"[{item['id']}] Q: {item['prompt']}\n  A: {ans}\n", flush=True)

    # --- Part 2: Ethiopian culture (English vs Amharic) ---
    print("\n=== PART 2: ETHIOPIAN EVERYDAY CULTURE (ENGLISH vs AMHARIC) ===", flush=True)
    ethiopian_out = []
    for item in ETHIOPIAN_CULTURE_PROMPTS:
        ans_en = generate_one(model, tokenizer, item["en"], max_new_tokens=180)
        ans_am = generate_one(model, tokenizer, item["am"], max_new_tokens=220)
        ans_am_en = translate_to_english(model, tokenizer, ans_am, "Amharic")
        ethiopian_out.append(
            {
                **item,
                "response_en": ans_en,
                "response_am": ans_am,
                "response_am_translated_en": ans_am_en,
            }
        )
        print(
            f"[{item['id']}] Gold: {item['gold_fact']}\n"
            f"  EN: {ans_en}\n"
            f"  AM: {ans_am}\n"
            f"  AM->EN: {ans_am_en}\n",
            flush=True,
        )

    # --- Part 3: Welsh culture (English vs Welsh) ---
    print("\n=== PART 3: WELSH EVERYDAY CULTURE (ENGLISH vs WELSH) ===", flush=True)
    welsh_out = []
    for item in WELSH_CULTURE_PROMPTS:
        ans_en = generate_one(model, tokenizer, item["en"], max_new_tokens=180)
        ans_cy = generate_one(model, tokenizer, item["cy"], max_new_tokens=200)
        ans_cy_en = translate_to_english(model, tokenizer, ans_cy, "Welsh")
        welsh_out.append(
            {
                **item,
                "response_en": ans_en,
                "response_cy": ans_cy,
                "response_cy_translated_en": ans_cy_en,
            }
        )
        print(
            f"[{item['id']}] Gold: {item['gold_fact']}\n"
            f"  EN: {ans_en}\n"
            f"  CY: {ans_cy}\n"
            f"  CY->EN: {ans_cy_en}\n",
            flush=True,
        )

    out = {
        "model": args.model,
        "us_default_norms": us_default_out,
        "ethiopian_culture": ethiopian_out,
        "welsh_culture": welsh_out,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[culture] Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

