import json, os, re
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
qdrant = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
    check_compatibility=False
)
client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY')
)

EXTRACTION_MODEL = 'openrouter/free'

SYSTEM_PROMPT = '''You are a financial KPI extraction model. Extract metrics from SEC filing chunks.
You MUST respond ONLY with this exact JSON format and nothing else:

If data IS present:
{"metric": "Revenue", "value": 117154, "unit": "million USD", "period": "Q1 FY2023", "yoy_change": "-5.5%", "source_quote": "Total net sales 117,154", "confidence": "HIGH"}

If data is NOT present:
{"confidence": "UNVERIFIABLE", "reason": "metric not found in text"}

Rules:
- value must be a number (no commas, no dollar signs)
- confidence must be exactly HIGH or UNVERIFIABLE
- never add extra fields
- never wrap in markdown
- always close the JSON properly with }'''

KPI_QUESTIONS = [
    'What was total revenue or net sales and its YoY change?',
    'What was gross margin or gross profit?',
    'What was operating income or EBIT?',
    'What was EPS (earnings per share) diluted?',
    'What forward guidance or outlook was provided?',
    'What was free cash flow or cash from operations?'
]

FINANCIAL_KEYWORDS = [
    'net sales', 'revenue', 'gross margin', 'operating income',
    'net income', 'earnings per share', 'cash flow', 'total sales'
]

def is_clean(text):
    if not text or len(text) < 50:
        return False
    text_lower = text.lower()
    if any(kw in text_lower for kw in FINANCIAL_KEYWORDS):
        return True
    words = text.split()
    xbrl = sum(1 for w in words if ':' in w or re.match(r'^\d{4}-\d{2}-\d{2}$', w))
    return (xbrl / len(words)) < 0.08

def clean_json(raw):
    raw = raw.strip()
    if raw.startswith('`'):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:-1]).strip()
    if raw.startswith('') and raw.endswith(''):
        raw = raw.strip('')
    raw = raw.strip()
    if raw.startswith('{') and not raw.endswith('}'):
        raw = raw + '}'
    return raw

def retrieve_chunks(ticker, query, top_k=20):
    vec = embedder.encode(query, normalize_embeddings=True).tolist()
    results = qdrant.query_points(
        collection_name='filings',
        query=vec,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key='ticker',
                match=models.MatchValue(value=ticker)
            )]
        ),
        limit=top_k
    )
    clean = []
    for r in results.points:
        text = r.payload.get('text', '')
        if is_clean(text):
            clean.append({'chunk_id': r.payload['chunk_id'], 'text': text})
        if len(clean) >= 5:
            break
    return clean

def extract_kpi(chunk_text, question):
    try:
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'Filing chunk:\n{chunk_text}\n\nExtract: {question}'}
            ],
            temperature=0,
            timeout=30,
            max_tokens=200
        )
        raw = resp.choices[0].message.content
        cleaned = clean_json(raw)
        parsed = json.loads(cleaned)

        if 'confidence' not in parsed:
            value = (parsed.get('value') or parsed.get('total_revenue') or
                     parsed.get('amount') or parsed.get('revenue'))
            metric = parsed.get('metric') or question.split('was ')[-1].split('?')[0].strip()
            if value is not None:
                parsed = {
                    'metric': metric,
                    'value': value,
                    'unit': parsed.get('unit', 'million USD'),
                    'period': parsed.get('period', ''),
                    'yoy_change': parsed.get('yoy_change', ''),
                    'source_quote': parsed.get('source_quote', ''),
                    'confidence': 'HIGH'
                }
            else:
                return None
        return parsed
    except Exception as e:
        return None

def extraction_agent(state: dict) -> dict:
    ticker = state['ticker']
    all_kpis, all_chunks = [], []

    for question in KPI_QUESTIONS:
        chunks = retrieve_chunks(ticker, f'{ticker} {question}')
        all_chunks.extend(chunks)
        for chunk in chunks:
            kpi = extract_kpi(chunk['text'], question)
            if kpi and kpi.get('confidence') not in ('UNVERIFIABLE', None):
                kpi['chunk_id'] = chunk['chunk_id']
                kpi['question'] = question
                all_kpis.append(kpi)
                break

    return {**state, 'raw_chunks': all_chunks, 'extracted_kpis': all_kpis}
