from dotenv import load_dotenv
import google.generativeai as genai
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List
import os

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


class QueryVariations(BaseModel):
    queries: List[str]
    

# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────
# Initialize Gemini
genai.configure(api_key=os.getenv("gemini"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")



prompt = f"""
Generate 3 different variations of this query that help retrieve relevant documents.

Original query: "{original_query}"

Your output MUST be valid JSON in this exact format:

{{
  "queries": [
    "variation 1",
    "variation 2",
    "variation 3"
  ]
}}

Do NOT add explanations, notes, or text outside the JSON.
"""

# Call Gemini
response = gemini_model.generate_content(prompt)

# Extract text
raw_output = response.text

print("Raw model output:")
print(raw_output)
clean_output = raw_output.strip().lstrip("```json").rstrip("```").strip()

# Parse JSON safely
try:
    data = json.loads(clean_output)
    variations = QueryVariations(**data).queries
except Exception as e:
    print("\n❌ JSON parsing failed, model did not return proper JSON.")
    print("Error:", e)
    raise SystemExit

print("\nGenerated Query Variations:")
for i, v in enumerate(variations, 1):
    print(f"{i}. {v}")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")