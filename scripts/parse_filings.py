import os, json, re, html
from pathlib import Path
from tqdm import tqdm
from collections import Counter

MAX_WORDS_PER_TABLE = 600

KEY_TERMS = [
    'net sales', 'revenue', 'gross margin', 'gross profit',
    'operating income', 'net income', 'earnings per share',
    'free cash flow', 'operating expenses', 'cost of sales',
    'ebitda', 'diluted', 'basic', 'total assets', 'cash flow'
]

def extract_primary_document(raw):
    blocks = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', raw, re.DOTALL)
    for block in blocks:
        t = re.search(r'<TYPE>([^\n]+)', block)
        if not t:
            continue
        if t.group(1).strip() in ('10-K', '10-Q', '10-K/A', '10-Q/A'):
            m = re.search(r'<TEXT>(.*?)(?:</TEXT>|$)', block, re.DOTALL)
            if m:
                return m.group(1)
    return ''

def extract_financial_tables(doc):
    tables = re.findall(r'<table[^>]*>(.*?)</table>', doc, re.DOTALL|re.IGNORECASE)
    financial = []
    for table in tables:
        text = re.sub(r'<[^>]+>', ' ', table)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 100:
            continue
        text_lower = text.lower()
        matches = sum(1 for term in KEY_TERMS if term in text_lower)
        if matches >= 2:
            financial.append(text)
    return financial

def extract_narrative_spans(doc):
    # Also extract readable narrative paragraphs
    tag_pattern = re.compile(r'<(p|div|span)[^>]*>(.*?)</\1>', re.DOTALL|re.IGNORECASE)
    sentences = []
    prev = ''
    for match in tag_pattern.finditer(doc):
        inner = match.group(2)
        inner = re.sub(r'<[^>]+>', ' ', inner)
        inner = html.unescape(inner)
        inner = re.sub(r'\s+', ' ', inner).strip()
        if len(inner) > 80 and inner != prev:
            sentences.append(inner)
            prev = inner
    text = ' '.join(sentences)
    # Split into paragraphs
    paragraphs = re.split(r'(?<=[.!?])\s{2,}', text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 100]

def extract_ticker(filepath):
    parts = Path(filepath).parts
    for i, p in enumerate(parts):
        if p == 'sec-edgar-filings' and i+1 < len(parts):
            return parts[i+1]
    return 'UNKNOWN'

def extract_form_type(filepath):
    for p in Path(filepath).parts:
        if p in ('10-K', '10-Q', '8-K'):
            return p
    return 'UNKNOWN'

def make_chunks(text, ticker, form_type, source_file, prefix, chunk_size=500, overlap=50):
    words = text.split()[:MAX_WORDS_PER_TABLE]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if len(chunk.strip()) < 60:
            continue
        chunks.append({
            'chunk_id': f'{ticker}_{form_type}_{Path(source_file).parent.name}_{prefix}_{i}',
            'ticker': ticker,
            'form_type': form_type,
            'source_file': str(source_file),
            'chunk_type': prefix,
            'text': chunk
        })
    return chunks

def process_filings(raw_dir='data/raw', out_dir='data/processed'):
    os.makedirs(out_dir, exist_ok=True)
    all_records = []
    skipped = 0
    files = list(Path(raw_dir).rglob('full-submission.txt'))
    print(f'Found {len(files)} filing files')

    for filepath in tqdm(files):
        ticker = extract_ticker(filepath)
        form_type = extract_form_type(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
        except Exception as e:
            skipped += 1
            continue

        doc = extract_primary_document(raw)
        if not doc:
            skipped += 1
            continue

        # Extract financial tables (priority)
        tables = extract_financial_tables(doc)
        for ti, table_text in enumerate(tables):
            chunks = make_chunks(table_text, ticker, form_type, filepath, f'table{ti}')
            all_records.extend(chunks)

        # Extract narrative paragraphs (secondary)
        paragraphs = extract_narrative_spans(doc)
        narrative_text = ' '.join(paragraphs[:50])  # cap narrative
        if narrative_text:
            chunks = make_chunks(narrative_text, ticker, form_type, filepath, 'narrative')
            all_records.extend(chunks)

    print(f'\nTotal chunks: {len(all_records)} | Skipped: {skipped}')
    print(f'\nPer ticker breakdown:')
    ticker_counts = Counter(r['ticker'] for r in all_records)
    for t, c in sorted(ticker_counts.items()):
        print(f'  {t}: {c} chunks')

    # Show chunk type breakdown
    type_counts = Counter(r['chunk_type'].replace('table0','table').replace('table1','table') for r in all_records)
    table_chunks = sum(v for k,v in type_counts.items() if 'table' in k)
    narrative_chunks = sum(v for k,v in type_counts.items() if 'narrative' in k)
    print(f'\nTable chunks: {table_chunks}')
    print(f'Narrative chunks: {narrative_chunks}')

    out_path = f'{out_dir}/chunks.json'
    with open(out_path, 'w') as f:
        json.dump(all_records, f)
    print(f'\nSaved {len(all_records)} chunks to {out_path}')

if __name__ == '__main__':
    process_filings()
