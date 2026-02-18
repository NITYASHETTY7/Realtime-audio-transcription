import os
import json
import psycopg2
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
client = genai.Client(api_key=GEMINI_API_KEY)

def analyze_conversation(text):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=text,
            config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": [
                                "Positive",
                                "Neutral",
                                "Negative",
                                "Agitated"
                            ]
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "Machine Operation Issues",
                                "Maintenance & Parts",
                                "Technical Troubleshooting"
                            ]
                        },
                        "search_query": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "sentiment",
                        "category",
                        "search_query"
                    ]
                }
            }
        )

        return response.parsed  # Direct dict, no json.loads()
    except Exception as e:
        print("Gemini schema error:", e)
        return None

def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 768}
    )
    return response.embeddings[0].values

def search_manuals(query):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    embedding = get_embedding(query)
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

def generate_solution_card(user_query, search_results):
    context = "\n\n".join(
        f"Section: {r[1]} (Pages {r[2]}-{r[3]})\n{r[0]}"
        for r in search_results
    )
    prompt = f"""
You are a technical support assistant.
User Problem:
{user_query}
Relevant Manual Content:
{context}
Create a concise solution card.
Keep:
- Cause: 2-3 sentences
- Solution: 4-5 short clear steps
- Clear professional tone
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()
