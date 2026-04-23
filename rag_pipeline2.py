# Load necessary Libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import torch

# Load Corpus (docs)
df = pd.read_csv("rag_docs/corpus_2025.csv")
texts = df["text"].fillna("").tolist()


# List of Questions
questions = [
    "What was Marquette's hitting percentage in 2025?",
    "Which Big East team had the most digs?",
    "List the setters on Creighton’s roster.",
    "Who were the best five hitters in the Big Ten, based on kills and hitting percentage?",
    "Who is likely to win if Kentucky and Florida play each other?",
    "Which player in the Big 12 had the most kills?",
    "Which team(s) ended the season with the best winning record?",
    "What was Stanford's blocking average?",
    "How many service aces did Texas A&M have?",
    "How did Wisconsin perform in five-set matches?",
    "Which Summit League tea had the highest hitting efficiency?",
    "Rank the ACC teams by total blocks.",
    "List the liberos on Louisville's roster",
    "Who was the starting middle blocker for Marquette?",
    "Who were the top three servers in the ACC based on aces?",
    "Compare the hitting percentages of the top hitters from Baylor and BYU.",
    "How did the top Big 12 setter compare to the top Big Ten setter?",
    "If Stanford played Nebraska, which team would statistically have the advantage?",
    "Based on 2025 stats, who would likely win between Pitt and Louisville?",
    "Which team would be favored in a matchup between Texas and Wisconsin?",
    "Which players most likely were named to the 2025 All‑American First Team?"
]


# Chunking
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]

kb_chunks = []
for t in texts:
    kb_chunks.extend(chunk_text(t))

print("Total chunks:", len(kb_chunks))



# Embeddings + faiss
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

kb_embeddings = embed_model.encode(
    kb_chunks,
    convert_to_numpy=True,
    show_progress_bar=True
).astype("float32")

dim = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(kb_embeddings)

print("FAISS index size:", index.ntotal)

