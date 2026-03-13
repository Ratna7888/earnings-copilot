# 📊 Earnings Intelligence Copilot

> Citation-grounded investment memos from SEC filings — Multi-agent LangGraph pipeline with verified KPIs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://earnings-copilot-almkl83oeuegfhkf9mcyci.streamlit.app/)

## 🚀 Live Demo

**[earnings-copilot-almkl83oeuegfhkf9mcyci.streamlit.app](https://earnings-copilot-almkl83oeuegfhkf9mcyci.streamlit.app/)**

Select any S&P 500 ticker, choose a fiscal period, and get a fully citation-grounded investment memo in seconds — with every number verified against the original SEC filing.

---

## 🧠 What It Does

Most LLM-based financial tools hallucinate numbers. This system doesn't.

Every KPI in the investment memo is:
1. **Retrieved** from actual SEC 10-K/10-Q filings via semantic search
2. **Extracted** using a structured JSON schema with source citations
3. **Verified** — the exact number is confirmed present in the source chunk before it enters the memo
4. Only **verified KPIs** make it into the final investment memo

---

## 🏗️ Architecture

**Data Pipeline:**
```
SEC EDGAR (220 filings) → parse_filings.py → 2,464 chunks → Qdrant Cloud
```

**Agent Pipeline:**

| Step | Agent | What it does |
|---|---|---|
| 1 | 🔍 Extraction Agent | Semantic search on Qdrant → LLM extracts KPIs as JSON |
| 2 | 📢 Narrative Agent | Generates risk flags and warnings |
| 3 | ✅ Verification Agent | Every number cross-checked against source chunk |
| 4 | 📝 Memo Writer Agent | Generates Bull/Bear/Verdict using only verified KPIs |
---

## 🤖 Fine-Tuned Models

Two models were fine-tuned for structured KPI extraction from SEC filings:

| Model | Base | Method | Size | Link |
|---|---|---|---|---|
| Mistral-7B adapter | Mistral-7B-Instruct-v0.3 | QLoRA | 27MB adapter | [🤗 HuggingFace](https://huggingface.co/ratnasekhar/earnings-copilot-mistral-7b) |
| Phi-3.5-mini merged | Phi-3.5-mini-instruct | QLoRA + merged | 7.6GB | [🤗 HuggingFace](https://huggingface.co/ratnasekhar/earnings-copilot-phi3-merged) |

Both models were trained to:
- Extract KPIs as structured JSON with mandatory source citations
- Output {"confidence": "UNVERIFIABLE"} instead of hallucinating when data is absent
- Trained on **619 balanced examples** (50% HIGH confidence, 50% UNVERIFIABLE)

---

## 📊 Results

| Metric | Value |
|---|---|
| Tickers covered | 20 S&P 500 companies |
| Filings ingested | 220 (10-K + 10-Q, 2020-2024) |
| Chunks indexed | 2,464 financial table chunks |
| Training examples | 619 balanced examples |
| Verification score | Up to 100% on major tickers |
| Fine-tuning time | ~75 min on T4 GPU |
| Adapter size | 27MB (vs 14.5GB full model) |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Vector database | Qdrant Cloud |
| Embeddings | BGE-Small-EN-v1.5 (local) |
| KPI extraction | OpenRouter free models |
| Memo writing | OpenRouter free models |
| Fine-tuning | QLoRA (PEFT + BitsAndBytes) |
| UI | Streamlit |
| Live stock prices | yfinance |
| Data source | SEC EDGAR API |

**Total infrastructure cost: .00** — built entirely on free tiers.

---

## 📁 Project Structure

| File | Purpose |
|---|---|
| `agents/state.py` | Shared TypedDict state passed between all agents |
| `agents/extraction_agent.py` | Qdrant retrieval + KPI extraction |
| `agents/narrative_agent.py` | Risk flagging |
| `agents/verification_agent.py` | Number cross-checking against source |
| `agents/graph.py` | LangGraph orchestration + memo writer |
| `scripts/download_filings.py` | SEC EDGAR ingestion |
| `scripts/parse_filings.py` | HTML/XBRL parsing → financial table chunks |
| `scripts/build_vectorstore.py` | Qdrant Cloud indexing |
| `scripts/generate_training_data.py` | LLM-assisted training data labeling |
| `scripts/balance_dataset.py` | Fix class imbalance (93% → 50% UNVERIFIABLE) |
| `scripts/format_for_finetuning.py` | ChatML formatting + train/val split |
| `ui/app.py` | Streamlit UI with live stock prices |
---

## 🚀 Run Locally
`ash
git clone https://github.com/Ratna7888/earnings-copilot
cd earnings-copilot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
`

Create .env:
`
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
`
`ash
streamlit run ui/app.py
`

---

## 🔬 Fine-Tuning Details

### Dataset Construction
- Downloaded 220 SEC filings from EDGAR for 20 S&P 500 companies
- Parsed financial table chunks using custom HTML/XBRL extractor
- Generated 3,084 training examples using LLM-assisted labeling
- Identified and fixed 93% class imbalance → resampled to 50/50

### QLoRA Configuration
`python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
`

### Before vs After Fine-tuning
| Behavior | Base Model | Fine-tuned |
|---|---|---|
| JSON schema | Inconsistent | Always correct |
| Hallucination | Invents numbers | Returns UNVERIFIABLE |
| Source citation | Missing | Always present |
| Refusal behavior | Rare | Trained explicitly |

---

## 📈 Example Output

**Input:** AAPL, Q1 2024

**Verified KPIs extracted:**
- Revenue: ,575M (Q1 FY2024) ✅
- Gross Margin: ,855M ✅  
- Operating Income: ,016M ✅
- Diluted EPS: .18 ✅

**Investment Memo:** Bull thesis, bear risks, and verdict — all grounded in verified filing data.

---

## 🤝 Acknowledgements

- [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) for free filing access
- [Qdrant](https://qdrant.tech/) for vector database free tier
- [OpenRouter](https://openrouter.ai/) for free LLM inference
- [HuggingFace](https://huggingface.co/) for model hosting
- [Streamlit](https://streamlit.io/) for deployment
