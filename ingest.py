import re
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if len(re.findall(r"\.\s*\.\s*\.", line)) >= 3:
            continue
        if re.fullmatch(r"\s*\d+\s*", line):
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


SECTION_HEADER = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,80})$", re.MULTILINE)


def split_by_sections(text):
    """Return list of (section_title, section_text)."""
    matches = list(SECTION_HEADER.finditer(text))
    if not matches:
        return [("(no section)", text)]

    sections = []
    # Preamble before the first header
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(("Preamble", preamble))

    for i, m in enumerate(matches):
        title = f"{m.group(1)} {m.group(2).strip()}"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def build_index(pdf_path, collection_name="docs"):
    raw_text = extract_text(pdf_path)
    text = clean_text(raw_text)
    print(f"Raw: {len(raw_text)} → Cleaned: {len(text)}")

    sections = split_by_sections(text)
    print(f"Detected {len(sections)} sections")

    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks, metas = [], []
    for title, body in sections:
        pieces = sub_splitter.split_text(body) if len(body) > 700 else [body]
        for j, piece in enumerate(pieces):
            if len(piece.split()) < 20:
                continue
            chunks.append(piece)
            metas.append({"section": title, "source": pdf_path, "piece": j})

    print(f"Produced {len(chunks)} chunks from {len(sections)} sections")
    for i, (c, m) in enumerate(zip(chunks[:8], metas[:8])):
        print(f"  [{i}] {m['section']!r} | {c[:100]!r}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=metas,
    )
    print(f"Stored {len(chunks)} chunks")


if __name__ == "__main__":
    build_index("sample.pdf")