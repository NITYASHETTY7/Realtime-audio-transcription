import os
import re
import time
import json
import psycopg2
from datetime import date
from dotenv import load_dotenv
from google import genai
import fitz  # PyMuPDF

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

PDF_PATH = "backend/xnc-1-2-manual-2.pdf"
START_PAGE = 20        # 0-indexed
END_PAGE = 200          # exclusive
MAX_WORDS_PER_CHUNK = 350
MIN_CHUNK_LENGTH = 150
DELAY_SECONDS = 1.2

DAILY_QUOTA = 1500
QUOTA_SAFETY_BUFFER = 50
QUOTA_FILE = ".gemini_quota.json"
IMPORTANT_KEYWORDS = [
    "error", "alarm", "fault", "parameter", "axis", "reset",
    "homing", "speed", "motor", "limit", "gain", "calibration",
    "warning", "failure", "overload", "encoder"
]

def load_quota():
    today = str(date.today())
    if os.path.exists(QUOTA_FILE):
        with open(QUOTA_FILE) as f:
            data = json.load(f)
        if data.get("date") == today:
            return data
    return {"date": today, "used": 0}
def save_quota(data):
    with open(QUOTA_FILE, "w") as f:
        json.dump(data, f)
def quota_remaining(data):
    return (DAILY_QUOTA - QUOTA_SAFETY_BUFFER) - data["used"]

HEADING_RE = re.compile(
    r"^("
    r"(?:\d+[\.\d]*\s+[A-Z].{3,})"
    r"|(?:CHAPTER\s+\w+.*)"
    r"|(?:[A-Z][A-Z\s]{4,})"
    r"|(?:[A-Z]\.\s+[A-Z].{3,})"
    r")$"
)

def is_heading(line):
    return bool(HEADING_RE.match(line.strip()))
def clean_text(text):
    text = text.replace("\x00", "")
    text = re.sub(r"\.{3,}", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_chunks_with_metadata(doc, start_page, end_page):
    chunks = []
    current_words = []
    current_pages = []
    current_section = "Unknown Section"
    def flush(words, pages, section):
        text = " ".join(words).strip()
        if len(text) < MIN_CHUNK_LENGTH:
            return
        lower_text = text.lower()
        if not any(k in lower_text for k in IMPORTANT_KEYWORDS):
            return
        chunks.append({
            "content": text,
            "page_start": min(pages),
            "page_end": max(pages),
            "section": section,
        })

    for page_idx in range(start_page, end_page):
        page = doc[page_idx]
        page_num = page_idx + 1
        raw = page.get_text("text")
        cleaned = clean_text(raw)
        for line in cleaned.split("\n"):
            line = line.strip()
            if not line:
                continue
            if is_heading(line):
                if current_words:
                    flush(current_words, current_pages, current_section)
                    current_words = []
                    current_pages = []
                current_section = line
            words = line.split()
            if len(current_words) + len(words) >= MAX_WORDS_PER_CHUNK:
                flush(current_words, current_pages, current_section)
                current_words = []
                current_pages = []
            current_words.extend(words)
            current_pages.append(page_num)
    if current_words:
        flush(current_words, current_pages, current_section)
    return chunks

def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "output_dimensionality": 768  
        }
    )
    return response.embeddings[0].values

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS manual_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    page_start INTEGER,
    page_end INTEGER,
    section TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

def ensure_schema(cur):
    cur.execute(SCHEMA_SQL)

def main():
    quota = load_quota()
    remaining = quota_remaining(quota)
    print(f"Gemini quota today: {quota['used']} used / {DAILY_QUOTA}")
    print(f"Remaining usable requests: {remaining}\n")
    if remaining <= 0:
        print("Daily quota reached.")
        return
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    ensure_schema(cur)
    conn.commit()
    print("Clearing previous embeddings...")
    cur.execute("DELETE FROM manual_embeddings;")
    conn.commit()
    print("Opening PDF...")
    doc = fitz.open(PDF_PATH)
    print(f"Extracting pages {START_PAGE+1} to {END_PAGE}...\n")
    chunks = extract_chunks_with_metadata(doc, START_PAGE, END_PAGE)
    doc.close()
    print(f"Total relevant chunks found: {len(chunks)}\n")
    if len(chunks) > remaining:
        print(f"Quota allows only {remaining} embeddings today.")
        chunks = chunks[:remaining]
    inserted = 0
    skipped = 0
    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)} | "
              f"Pages {chunk['page_start']}-{chunk['page_end']} | "
              f"{chunk['section'][:40]}... ", end="")
        try:
            embedding = get_embedding(chunk["content"])
            cur.execute(
                """
                INSERT INTO manual_embeddings
                (content, embedding, page_start, page_end, section)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    chunk["content"],
                    embedding,
                    chunk["page_start"],
                    chunk["page_end"],
                    chunk["section"],
                ),
            )
            conn.commit()
            quota["used"] += 1
            save_quota(quota)
            inserted += 1
            print("OK")
        except Exception as e:
            conn.rollback()
            skipped += 1
            print(f"SKIPPED - {e}")
        time.sleep(DELAY_SECONDS)

    print("\nDone.")
    print(f"Inserted: {inserted}")
    print(f"Skipped: {skipped}")
    print(f"Quota used today: {quota['used']} / {DAILY_QUOTA}")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
