# Introduction to Language Adaptation — Lecture Demos & Evaluations

Reproducible scripts and results accompanying the **Introduction to Language Adaptation** lecture (COM3110/4115), evaluated with a single open-weights model (**Qwen3.5-9B**, zero-shot, `bfloat16`, non-thinking mode).

The repository is organised by lecture section:

1. **Section 1 (Background — Slide 9):** Cross-lingual task performance gap on **Belebele** (reading comprehension) and **FLORES+** (machine translation), plus tokeniser overfragmentation statistics.
2. **Section 5 (Discussion & Future Directions):** Cultural defaults (BLEnD fireworks prompts, US vs. UK norms), everyday cultural knowledge in English vs. Amharic/Welsh, and safety-critical localisation (emergency numbers).

---

## Setup

### Prerequisites
- **Python 3.10+**
- **GPU:**
  - `Qwen/Qwen3.5-9B` (BF16): ~24 GB+ VRAM (e.g. Colab A100, G4, or L4)
  - `Qwen/Qwen3.5-4B` (BF16): ~12 GB+ VRAM (e.g. RTX 3060 / Colab T4)

### Installation

```bash
chmod +x install.sh
./install.sh
source .venv/bin/activate
```

`install.sh` uses `uv` if available, falling back to `python3 -m venv` + `pip`.

---

## Section 1: Cross-Lingual Performance Gap (Slide 9)

Measures task performance and tokeniser overfragmentation across high- and low-resource languages on `Qwen/Qwen3.5-9B`:
- **Belebele** (`facebook/belebele`, `n = 900`): 4-way multiple-choice reading comprehension in English (`eng_Latn`), French (`fra_Latn`, reference), and Amharic (`amh_Ethi`), evaluated via next-token logits over `A`/`B`/`C`/`D` (`enable_thinking=False`).
- **FLORES+** (`openlanguagedata/flores_plus`, `devtest`, `n = 150`): Greedy-decoded machine translation (`chrF++` with `word_order=2` and `BLEU`) between English and French, Welsh (`cym_Latn`), and Amharic (`amh_Ethi`), plus average token and UTF-8 byte counts per sentence.

> **Hugging Face access for FLORES+:** `openlanguagedata/flores_plus` is gated (auto-approved). Visit [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus), accept the terms, and authenticate via `huggingface-cli login` (or `export HF_TOKEN=...`).

### Run & Plot

```bash
# Full evaluation (A100 / 40 GB+ GPU)
python eval_belebele_flores.py \
  --model Qwen/Qwen3.5-9B \
  --belebele-n 900 \
  --flores-n 150 \
  --belebele-bs 16 \
  --flores-bs 32 \
  --output eval_results.json

# Smaller GPU / quick run (12–24 GB VRAM)
python eval_belebele_flores.py \
  --model Qwen/Qwen3.5-4B \
  --belebele-n 200 \
  --flores-n 100 \
  --belebele-bs 4 \
  --flores-bs 8 \
  --output eval_results.json

# Generate the 16:9 Slide 9 figure (slide9_performance_gap.png and .svg)
python plot_results.py
```

### Results (`Qwen/Qwen3.5-9B`, NVIDIA A100)

Full outputs are saved in [`eval_results.json`](eval_results.json).

**Belebele (Reading Comprehension, `n = 900`)**

| Language | Code | Accuracy (%) | 95% CI | Avg. Prompt Tokens |
|---|---|---:|---:|---:|
| English | `eng_Latn` | **94.78** | ±1.45 | 207.3 (1.0×) |
| French *(ref.)* | `fra_Latn` | **92.44** | ±1.73 | 260.8 (1.3×) |
| Amharic | `amh_Ethi` | **68.11** | ±3.04 | 669.6 (3.2×) |

**FLORES+ (Machine Translation, `devtest`, `n = 150`)**

| Language | En → Target (`chrF++` / `BLEU`) | Target → En (`chrF++` / `BLEU`) | Avg. Tokens / Sent. | Avg. UTF-8 Bytes / Sent. |
|---|---:|---:|---:|---:|
| English | — | — | 28.0 (1.0×) | 133.3 |
| French *(ref.)* | **68.82** / 48.80 | **67.18** / 42.79 | 39.0 (1.4×) | 164.7 |
| Welsh (`cym_Latn`) | **51.11** / 25.47 | **71.43** / 51.07 | 56.2 (2.0×) | 140.8 |
| Amharic (`amh_Ethi`) | **18.36** / 2.49 | **49.74** / 24.03 | 119.8 (4.3×) | 223.4 |

---

## Section 5: Cultural Defaults & Safety-Critical Localisation

No gated dataset is required for these demos.

### Run

```bash
# Live 6-prompt classroom demo (BLEnD fireworks prompts + emergency number in English & Amharic)
python demo_live_culture_safety.py --model Qwen/Qwen3.5-9B --output live_demo_results.json

# Broader cultural evaluation (US-default norms + Ethiopian & Welsh everyday culture)
python demo_culture.py --model Qwen/Qwen3.5-9B --output culture_results.json
```

### Key Findings (`Qwen/Qwen3.5-9B`)

- **BLEnD Fireworks Prompts ([`live_demo_results.json`](live_demo_results.json)):**
  - *No country (EN):* Leads with **July 4th (US)** and omits UK Bonfire Night (5 November).
  - *UK (EN):* Picks **New Year's Eve (31 December)** as the primary answer, mentioning Bonfire Night only as a secondary occasion.
  - *Ethiopia (Amharic, BLEnD):* Fails to recognise `ርችት` (*fireworks*), transliterating it as *"Ritcht"* and claiming it is not a recognised term in Ethiopian law.
  - *Ethiopia (EN):* Recovers **New Year's Day (*Enkutatash*)** (confirming BLEnD's finding that English outperforms local languages), but still assigns the Western date **January 1st** instead of 11 September (*Meskerem 1*).
- **Safety-Critical Emergency Number ([`live_demo_results.json`](live_demo_results.json)):**
  - *English:* Gives US **911**, UK **999 / 112**, Europe **112**, plus actionable CPR/AED steps.
  - *Amharic:* Misinterprets the breathing emergency as an emotional message and exhausts its 220-token budget (due to ~4× overfragmentation) before giving any emergency number — missing Ethiopia's **907** ambulance number.
- **Broader Cultural Evaluation ([`culture_results.json`](culture_results.json)):**
  - *US-default norms:* Defaults to May for Mother's Day (not UK March), 1 January for the tax year (not UK 6 April), US meaning of *pants*, and 911 first.
  - *Fluent language ≠ cultural knowledge:* Correctly answers Ethiopian calendar (*13 months, Pagumen*), Buna coffee ceremony rounds (*Abol, Tona, Baraka*), *Iddir* vs. *Equb*, *Dydd Santes Dwynwen* (25 January), *Mari Lwyd*, and *Calennig* in **English**, yet hallucinates basic facts when asked the exact same questions in **Amharic** or **Welsh**.

---

## Repository Files

| File | Section | Description |
|---|---|---|
| `eval_belebele_flores.py` | Section 1 (Slide 9) | Belebele & FLORES+ evaluation script |
| `plot_results.py` | Section 1 (Slide 9) | Generates 16:9 Slide 9 chart (`slide9_performance_gap.png`/`.svg`) |
| `eval_results.json` | Section 1 (Slide 9) | Saved Belebele & FLORES+ evaluation results |
| `slide9_performance_gap.{png,svg}` | Section 1 (Slide 9) | Slide 9 two-panel performance chart |
| `demo_live_culture_safety.py` | Section 5 | Live 6-prompt culture & emergency-number demo |
| `live_demo_results.json` | Section 5 | Saved outputs from `demo_live_culture_safety.py` |
| `demo_culture.py` | Section 5 | US-default norms & Ethiopian/Welsh cultural knowledge demo |
| `culture_results.json` | Section 5 | Saved outputs from `demo_culture.py` |
| `install.sh`, `requirements.txt` | Setup | One-step environment setup and dependencies |
