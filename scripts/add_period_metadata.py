"""Backfill fiscal_year / fiscal_quarter / period_end onto existing chunks and Qdrant
points without re-embedding. Point ids are chunk indexes (see build_vectorstore.py)."""
import json, os, sys
from collections import defaultdict
from qdrant_client import QdrantClient
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(__file__))
from parse_filings import extract_fiscal_period

load_dotenv()

with open('data/processed/chunks.json') as f:
    chunks = json.load(f)

periods = {}
for c in chunks:
    src = c['source_file'].replace('\\', '/')
    if src not in periods:
        with open(src, encoding='utf-8', errors='ignore') as f:
            periods[src] = extract_fiscal_period(f.read(5000), c['form_type'])
    c.update(periods[src])

client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))

# Make sure ids still line up with chunk order before writing anything
sample = client.retrieve('filings', ids=[0, len(chunks) // 2, len(chunks) - 1], with_payload=['chunk_id'])
assert all(p.payload['chunk_id'] == chunks[p.id]['chunk_id'] for p in sample), 'point ids do not match chunks.json'

by_period = defaultdict(list)
for i, c in enumerate(chunks):
    by_period[(c['period_end'], c['fiscal_year'], c['fiscal_quarter'])].append(i)
for (end, year, quarter), ids in by_period.items():
    client.set_payload('filings', payload={'period_end': end, 'fiscal_year': year, 'fiscal_quarter': quarter}, points=ids)

client.create_payload_index('filings', field_name='fiscal_year', field_schema='integer')
client.create_payload_index('filings', field_name='fiscal_quarter', field_schema='keyword')

with open('data/processed/chunks.json', 'w') as f:
    json.dump(chunks, f)
print(f'Tagged {len(chunks)} chunks across {len(by_period)} filings')
