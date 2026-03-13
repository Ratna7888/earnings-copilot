import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

# Load chunks
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# ✅ Qdrant Cloud
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Verify connection
try:
    cols = client.get_collections()
    print(f"✅ Connected to Qdrant Cloud. Existing collections: {[c.name for c in cols.collections]}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit()

# Create collection only if it doesn't exist
existing = [c.name for c in client.get_collections().collections]
if "filings" not in existing:
    client.create_collection(
        collection_name="filings",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("Collection 'filings' created.")
else:
    print("Collection 'filings' already exists, appending...")

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Embed and upsert in batches
BATCH = 64
for i in tqdm(range(0, len(chunks), BATCH)):
    batch = chunks[i:i+BATCH]
    texts = [c["text"] for c in batch]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    points = [
        PointStruct(
            id=i+j,
            vector=embeddings[j],
            payload={k: v for k, v in batch[j].items()}
        )
        for j in range(len(batch))
    ]
    client.upsert(collection_name="filings", points=points)

print(f"✅ Indexed {len(chunks)} chunks into Qdrant Cloud")