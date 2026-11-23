# Introduction to Language Adaptation — Cross-Lingual Evaluation

Reproducible evaluation scripts accompanying the lecture **Introduction to Language Adaptation** (COM3110/4115).

This repository measures the cross-lingual performance gap of a single open-weights LLM (**Qwen3.5-9B**, zero-shot, non-thinking mode) across high- and low-resource languages:

- **Belebele** (`facebook/belebele`): 4-way multiple-choice reading comprehension in **English** (`eng_Latn`), **French** (`fra_Latn`, high-resource reference), and **Amharic** (`amh_Ethi`).
- **FLORES+** (`openlanguagedata/flores_plus`, `devtest`): Machine translation (`chrF++` and `BLEU`) between **English** and **Welsh**, **Amharic**, and **French** (high-resource reference), along with tokenizer fertility statistics (average tokens and UTF-8 bytes per sentence).

## Prerequisites

- **Python 3.10+**
- **GPU**:
  - `Qwen/Qwen3.5-9B` (BF16): ~24 GB+ VRAM (e.g. Colab A100, G4, or L4)
  - `Qwen/Qwen3.5-4B` (BF16): ~12 GB+ VRAM (e.g. RTX 3060 / Colab T4)
- **Hugging Face access to FLORES+**:
  1. Visit [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus) and click *Agree and access repository* (auto-approved).
  2. Authenticate locally with `huggingface-cli login` (or set `export HF_TOKEN=...`).

## Installation

```bash
chmod +x install.sh
./install.sh
source .venv/bin/activate
```

`install.sh` automatically uses `uv` if installed, and falls back to `python3 -m venv` + `pip`.

## Running the Evaluation

### Standard run (`Qwen/Qwen3.5-9B`, A100 / 40GB+ GPU)

```bash
python eval_belebele_flores.py \
  --model Qwen/Qwen3.5-9B \
  --belebele-n 900 \
  --flores-n 150 \
  --belebele-bs 16 \
  --flores-bs 32 \
  --output eval_results.json
```

### Smaller GPU / quick run (e.g. 12–24 GB VRAM)

Use `Qwen/Qwen3.5-4B` and smaller batch sizes:

```bash
python eval_belebele_flores.py \
  --model Qwen/Qwen3.5-4B \
  --belebele-n 200 \
  --flores-n 100 \
  --belebele-bs 4 \
  --flores-bs 8 \
  --output eval_results.json
```

### Generating the Slide Chart

Once `eval_results.json` is produced, generate the two-panel figure (`slide9_performance_gap.png` and `.svg`):

```bash
python plot_results.py
```

### Section 5 Demos: Cultural Defaults & Safety-Critical Localization

Run the classroom demos for cultural defaults, Ethiopian/Welsh cultural knowledge, and safety-critical localization (no gated dataset required):

```bash
# Quick 6-prompt live demo (BLEnD fireworks prompts + emergency number in English & Amharic)
python demo_live_culture_safety.py --model Qwen/Qwen3.5-9B --output live_demo_results.json

# Broader cultural evaluation (US-default norms, Ethiopian culture in EN vs AM, Welsh culture in EN vs CY)
python demo_culture.py --model Qwen/Qwen3.5-9B --output culture_results.json
```

## Evaluation Details

- **Belebele**: Zero-shot multiple-choice evaluation using next-token logits over the option letters (`A`, `B`, `C`, `D`) with Qwen3.5 thinking mode disabled (`enable_thinking=False`).
- **FLORES+**: Zero-shot greedy decoding (`do_sample=False`, `max_new_tokens=160`) evaluated with `sacrebleu` `chrF++` (`word_order=2`) and `BLEU`.

## Results (`Qwen/Qwen3.5-9B`, Zero-Shot, Non-Thinking Mode)

Evaluated on an NVIDIA A100 (`bfloat16`, `enable_thinking=False`). Full outputs are saved in `eval_results.json`.

### Belebele (Reading Comprehension, `n = 900`)

| Language | Code | Accuracy (%) | 95% CI | Avg. Prompt Tokens |
|---|---|---:|---:|---:|
| English | `eng_Latn` | **94.78** | ±1.45 | 207.3 (1.0×) |
| French *(ref.)* | `fra_Latn` | **92.44** | ±1.73 | 260.8 (1.3×) |
| Amharic | `amh_Ethi` | **68.11** | ±3.04 | 669.6 (3.2×) |

### FLORES+ (Machine Translation, `devtest`, `n = 150`)

| Language | En → Target (chrF++ / BLEU) | Target → En (chrF++ / BLEU) | Avg. Tokens / Sent. | Avg. UTF-8 Bytes / Sent. |
|---|---:|---:|---:|---:|
| English | — | — | 28.0 (1.0×) | 133.3 |
| French *(ref.)* | **68.82** / 48.80 | **67.18** / 42.79 | 39.0 (1.4×) | 164.7 |
| Welsh (`cym_Latn`) | **51.11** / 25.47 | **71.43** / 51.07 | 56.2 (2.0×) | 140.8 |
| Amharic (`amh_Ethi`) | **18.36** / 2.49 | **49.74** / 24.03 | 119.8 (4.3×) | 223.4 |

### Section 5 Demos (`Qwen/Qwen3.5-9B`, Zero-Shot, Non-Thinking Mode)

- **Live culture & safety-critical demo (`demo_live_culture_safety.py`, `live_demo_results.json`)**:
  - *Fireworks (BLEnD)*: With no country given, leads with **July 4th (US)** and omits UK Bonfire Night; when explicitly asked about the **UK**, still picks **New Year's Eve (Dec 31)** over Bonfire Night (Nov 5); in **Amharic**, fails to recognize `ርችት` (*fireworks*) and transliterates it as *"Ritcht"*; in **English** specifying Ethiopia, recovers **New Year's Day (*Enkutatash*)** but assigns the Western date **January 1st** (instead of Sept 11 / Meskerem 1).
  - *Emergency number (safety-critical default)*: In **English**, gives US 911, UK 999/112, and CPR/AED steps; in **Amharic**, misinterprets the breathing emergency as an emotional message and runs out of its 220-token budget before giving any emergency number (missing Ethiopia's **907** ambulance number).
- **Broader cultural evaluation (`demo_culture.py`, `culture_results.json`)**:
  - *US-default norms*: Answered *"second Sunday of May in the United States"* for Mother's Day, *"January 1st"* for the tax year, *"American English ... waist to the ankles"* for *pants*, and *"911"* first for emergency numbers.
  - *Fluent language ≠ cultural knowledge*: Correctly answered Ethiopian calendar (*13 months, Pagumen*), Buna rounds (*Abol, Tona, Baraka*), *Iddir* vs *Equb*, *Dydd Santes Dwynwen* (25 Jan), *Mari Lwyd*, and *Calennig* when asked in **English**, yet hallucinated basic facts when asked the exact same questions in **Amharic** or **Welsh**.

## Files

- `eval_belebele_flores.py` — Main evaluation script for Belebele and FLORES+.
- `plot_results.py` — Plots the two-panel summary chart from `eval_results.json`.
- `demo_live_culture_safety.py` — Live 6-prompt demo (BLEnD fireworks + English/Amharic emergency number).
- `demo_culture.py` — US-default norms & Ethiopian/Welsh cultural knowledge demo.
- `install.sh` — One-step environment setup script.
- `requirements.txt` — Python dependencies.

