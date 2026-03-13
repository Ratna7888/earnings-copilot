import json, os, random, time
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY')
)

SYSTEM_PROMPT = '''You are a financial data extraction expert.
Given a chunk from a SEC filing, extract KPIs if present.
Respond ONLY with a single valid JSON object. No markdown, no backticks, no explanation.

If the number IS in the chunk:
{"metric": "Revenue", "value": 94.84, "unit": "billion USD", "period": "Q3 FY2024", "yoy_change": "+6.1%", "source_quote": "exact quote max 20 words", "confidence": "HIGH"}

If the number is NOT clearly in the chunk:
{"confidence": "UNVERIFIABLE", "reason": "brief reason"}'''

KPI_QUESTIONS = [
    'What was total revenue and its YoY change?',
    'What was gross margin or gross profit?',
    'What was operating income or EBIT?',
    'What was EPS (earnings per share)?',
    'What forward guidance was provided?',
    'What was free cash flow?'
]

def clean_json(raw):
    raw = raw.strip()
    if raw.startswith('`'):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:-1])
    return raw.strip().strip('')

def call_llm(user):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model='openrouter/free',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user}
                ],
                temperature=0,
                timeout=30
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if '429' in err:
                wait = 60 * (attempt + 1)
                print(f'\nRate limited — waiting {wait}s before retry {attempt+1}/3')
                time.sleep(wait)
            elif '401' in err:
                print('\nInvalid OpenRouter API key')
                return None
            else:
                print(f'\nError: {err[:100]}')
                time.sleep(5)
    print('\nAll retries exhausted — stopping')
    return None

with open('data/processed/chunks.json') as f:
    all_chunks = json.load(f)

os.makedirs('data/finetune', exist_ok=True)
out_path = 'data/finetune/raw_training_data.json'
sample_path = 'data/finetune/sampled_chunk_ids.json'

# Load or create fixed sample (same 500 chunks every run)
if os.path.exists(sample_path):
    with open(sample_path) as f:
        sampled_ids = set(json.load(f))
    sample = [c for c in all_chunks if c['chunk_id'] in sampled_ids]
    print(f'Loaded fixed sample of {len(sample)} chunks')
else:
    sample = random.sample(all_chunks, min(500, len(all_chunks)))
    sampled_ids = [c['chunk_id'] for c in sample]
    with open(sample_path, 'w') as f:
        json.dump(sampled_ids, f)
    print(f'Created fixed sample of {len(sample)} chunks')

# Load existing results (resume support)
if os.path.exists(out_path):
    with open(out_path) as f:
        training_data = json.load(f)
    done_ids = {(d['chunk_id'], d['question']) for d in training_data}
    remaining = len(sample) * 6 - len(done_ids)
    print(f'Resuming — {len(training_data)} done, {remaining} remaining')
else:
    training_data = []
    done_ids = set()

# Main loop
stopped = False
for chunk in tqdm(sample):
    if stopped:
        break
    for question in KPI_QUESTIONS:
        if (chunk['chunk_id'], question) in done_ids:
            continue
        raw = call_llm(f"Chunk:\n{chunk['text']}\n\nQuestion: {question}")
        if raw is None:
            stopped = True
            break
        try:
            output = clean_json(raw)
            json.loads(output)  # validate
            training_data.append({
                'chunk_id': chunk['chunk_id'],
                'question': question,
                'chunk_text': chunk['text'],
                'output': output
            })
            if len(training_data) % 50 == 0:
                with open(out_path, 'w') as f:
                    json.dump(training_data, f, indent=2)
                print(f'\nCheckpoint: {len(training_data)} examples saved')
        except Exception:
            continue

with open(out_path, 'w') as f:
    json.dump(training_data, f, indent=2)
print(f'\nTotal: {len(training_data)} / {len(sample)*6} possible ({len(training_data)/(len(sample)*6)*100:.1f}%)')
