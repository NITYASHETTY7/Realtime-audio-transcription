import os
import psycopg2
from dotenv import load_dotenv
from google import genai

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────
def get_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 768}
    )
    return response.embeddings[0].values



# ─────────────────────────────────────────────────────────────
# Vector Search
# ─────────────────────────────────────────────────────────────
def search_manuals(query: str):
    print("\nEmbedding query...")
    embedding = get_embedding(query)

    print("Searching vector database...\n")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT content, section, page_start, page_end,
               embedding <=> %s::vector AS distance
        FROM manual_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
        """,
        (embedding, embedding),
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results


# ─────────────────────────────────────────────────────────────
# Solution Card Generator
# ─────────────────────────────────────────────────────────────
def generate_solution_card(user_query, search_results):
    context = "\n\n".join(
        f"Section: {r[1]} (Pages {r[2]}-{r[3]})\n{r[0]}"
        for r in search_results
    )

    prompt = f"""
You are a CNC technical support assistant.

User Problem:
{user_query}

Relevant Manual Extract:
{context}

Create a concise solution card.

Keep:
- Cause: 2-3 short sentences
- Solution: 4-5 clear steps
- Professional tone
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text.strip()


# ─────────────────────────────────────────────────────────────
# Main Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    user_query = input("Enter search query: ")

    results = search_manuals(user_query)

    print("\nTop Matches:\n")

    for i, r in enumerate(results, 1):
        print(f"Result {i}")
        print(f"Section: {r[1]}")
        print(f"Pages: {r[2]}-{r[3]}")
        print(f"Distance: {r[4]}")
        print(r[0][:500])  # print first 500 chars only
        print("-" * 60)

    print("\nGenerating solution card...\n")

    card = generate_solution_card(user_query, results)

    print("\n==============================")
    print("       SOLUTION CARD")
    print("==============================\n")
    print(card)
    print("\n================================")
