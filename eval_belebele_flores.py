#!/usr/bin/env python3
"""
Brief evaluation of Qwen3.5 on Belebele (reading comprehension) and FLORES+ (translation)
for English, Welsh, and Amharic (plus French as a high-resource translation reference).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


LANG_NAMES = {
    "eng_Latn": "English",
    "cym_Latn": "Welsh",
    "amh_Ethi": "Amharic",
    "fra_Latn": "French",
}


def load_model_and_tokenizer(model_id: str):
    print(f"[eval] Loading tokenizer and model: {model_id}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[eval] AutoModelForCausalLM failed ({e}); trying AutoModelForImageTextToText", flush=True)
        if AutoModelForImageTextToText is None:
            raise
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    print(f"[eval] Loaded in {time.time() - t0:.1f}s (dtype={dtype})", flush=True)
    return model, tokenizer


def format_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def eval_belebele(model, tokenizer, langs: list[str], n_samples: int, batch_size: int):
    results = {}
    choice_letters = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choice_letters]
    print(f"[belebele] Choice token IDs: {dict(zip(choice_letters, choice_ids))}", flush=True)

    for lang in langs:
        t0 = time.time()
        try:
            ds = load_dataset("facebook/belebele", lang, split="test")
        except ValueError as e:
            print(f"[belebele] SKIPPING {lang}: not available in facebook/belebele ({e})", flush=True)
            results[lang] = {
                "language": LANG_NAMES.get(lang, lang),
                "available": False,
                "note": "Not included in facebook/belebele",
            }
            continue
        if n_samples and n_samples < len(ds):
            ds = ds.select(range(n_samples))

        prompts = []
        gold_indices = []
        for ex in ds:
            user_text = (
                "Read the following passage and answer the question by choosing the correct option (A, B, C, or D). "
                "Respond with only a single letter: A, B, C, or D.\n\n"
                f"Passage: {ex['flores_passage']}\n\n"
                f"Question: {ex['question']}\n"
                f"A. {ex['mc_answer1']}\n"
                f"B. {ex['mc_answer2']}\n"
                f"C. {ex['mc_answer3']}\n"
                f"D. {ex['mc_answer4']}\n\n"
                "Answer:"
            )
            prompts.append(format_chat(tokenizer, user_text))
            gold_indices.append(int(ex["correct_answer_num"]) - 1)

        preds = []
        total_tokens = 0
        device = next(model.parameters()).device

        with torch.inference_mode():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(device)
                total_tokens += int(enc["attention_mask"].sum().item())
                out = model(**enc, use_cache=False, logits_to_keep=1)
                # Next-token logits at the last position (left-padded -> index -1)
                last_logits = out.logits[:, -1, choice_ids]
                batch_preds = last_logits.argmax(dim=-1).tolist()
                preds.extend(batch_preds)
        torch.cuda.empty_cache()

        correct = sum(int(p == g) for p, g in zip(preds, gold_indices))
        n = len(gold_indices)
        acc = correct / n
        se = math.sqrt(acc * (1 - acc) / n)
        elapsed = time.time() - t0
        results[lang] = {
            "language": LANG_NAMES.get(lang, lang),
            "n": n,
            "correct": correct,
            "accuracy": round(acc * 100, 2),
            "se": round(se * 100, 2),
            "avg_prompt_tokens": round(total_tokens / n, 1),
            "elapsed_s": round(elapsed, 1),
        }
        print(
            f"[belebele] {lang} ({LANG_NAMES.get(lang, lang)}): "
            f"{acc*100:.2f}% ± {1.96*se*100:.2f}% (n={n}, avg_tok={total_tokens/n:.1f}, {elapsed:.1f}s)",
            flush=True,
        )
    return results


def load_flores_devtest(lang: str, n_samples: int) -> list[str]:
    ds = load_dataset("openlanguagedata/flores_plus", lang, split="devtest")
    if n_samples and n_samples < len(ds):
        ds = ds.select(range(n_samples))
    return [ex["text"] for ex in ds]


def clean_translation(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def eval_flores(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
):
    # Preload unique languages and compute token counts (overfragmentation stats)
    unique_langs = sorted({l for pair in pairs for l in pair})
    texts = {lang: load_flores_devtest(lang, n_samples) for lang in unique_langs}

    token_stats = {}
    for lang, sents in texts.items():
        tok_lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sents]
        char_lens = [len(s) for s in sents]
        byte_lens = [len(s.encode("utf-8")) for s in sents]
        token_stats[lang] = {
            "language": LANG_NAMES.get(lang, lang),
            "n": len(sents),
            "avg_tokens": round(sum(tok_lens) / len(tok_lens), 2),
            "avg_chars": round(sum(char_lens) / len(char_lens), 2),
            "avg_utf8_bytes": round(sum(byte_lens) / len(byte_lens), 2),
            "tokens_per_char": round(sum(tok_lens) / sum(char_lens), 3),
        }
        print(
            f"[flores+ tok] {lang}: avg_tokens={token_stats[lang]['avg_tokens']}, "
            f"avg_chars={token_stats[lang]['avg_chars']}, avg_bytes={token_stats[lang]['avg_utf8_bytes']}",
            flush=True,
        )

    chrf_metric = CHRF(word_order=2)  # chrF++
    bleu_metric = BLEU()
    device = next(model.parameters()).device

    pair_results = {}
    for src_lang, tgt_lang in pairs:
        t0 = time.time()
        src_sents = texts[src_lang]
        ref_sents = texts[tgt_lang]
        src_name = LANG_NAMES.get(src_lang, src_lang)
        tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)

        prompts = [
            format_chat(
                tokenizer,
                f"Translate the following sentence from {src_name} to {tgt_name}. "
                f"Output only the translation, with no extra explanation.\n\n{s}",
            )
            for s in src_sents
        ]

        hyps = []
        with torch.inference_mode():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)
                in_len = enc["input_ids"].shape[1]
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for row in gen:
                    decoded = tokenizer.decode(row[in_len:], skip_special_tokens=True)
                    hyps.append(clean_translation(decoded))

        chrf_score = chrf_metric.corpus_score(hyps, [ref_sents]).score
        bleu_score = bleu_metric.corpus_score(hyps, [ref_sents]).score
        elapsed = time.time() - t0
        key = f"{src_lang}->{tgt_lang}"
        pair_results[key] = {
            "src": src_name,
            "tgt": tgt_name,
            "n": len(src_sents),
            "chrf++": round(chrf_score, 2),
            "bleu": round(bleu_score, 2),
            "elapsed_s": round(elapsed, 1),
            "samples": [
                {"src": src_sents[j], "ref": ref_sents[j], "hyp": hyps[j]}
                for j in range(min(3, len(src_sents)))
            ],
        }
        print(
            f"[flores+] {key} ({src_name}->{tgt_name}): "
            f"chrF++={chrf_score:.2f}, BLEU={bleu_score:.2f} (n={len(src_sents)}, {elapsed:.1f}s)",
            flush=True,
        )
        torch.cuda.empty_cache()

    return {"token_stats": token_stats, "translations": pair_results}


def main():
    hf_token_path = Path("/content/.hf_token")
    if hf_token_path.exists() and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token_path.read_text().strip()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--belebele-n", type=int, default=900)
    parser.add_argument("--flores-n", type=int, default=200)
    parser.add_argument("--belebele-bs", type=int, default=8)
    parser.add_argument("--flores-bs", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", default="/content/eval_results.json")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    belebele_langs = ["eng_Latn", "fra_Latn", "amh_Ethi"]
    flores_pairs = [
        ("eng_Latn", "cym_Latn"),
        ("cym_Latn", "eng_Latn"),
        ("eng_Latn", "amh_Ethi"),
        ("amh_Ethi", "eng_Latn"),
        ("eng_Latn", "fra_Latn"),
        ("fra_Latn", "eng_Latn"),
    ]

    belebele_res = eval_belebele(model, tokenizer, belebele_langs, args.belebele_n, args.belebele_bs)
    flores_res = eval_flores(
        model,
        tokenizer,
        flores_pairs,
        args.flores_n,
        args.flores_bs,
        args.max_new_tokens,
    )

    out = {
        "model": args.model,
        "belebele_n": args.belebele_n,
        "flores_n": args.flores_n,
        "belebele": belebele_res,
        "flores_plus": flores_res,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[eval] Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

