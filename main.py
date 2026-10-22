from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pymupdf
from docx import Document
import chromadb
import uuid
import io
import json
import os
app = FastAPI(title="Multi-Capability Assistant")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="uploaded_documents"
)

DOCUMENTS_FILE = "documents.json"
def load_documents():
    if os.path.exists(DOCUMENTS_FILE):
        try:
            with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []
documents = load_documents()
def extract_text(filename, content):
    # Extract text from PDF
    if filename.lower().endswith(".pdf"):
        try:
            with pymupdf.open(
                stream=content,
                filetype="pdf"
            ) as pdf:
                return "\n".join(
                    page.get_text() for page in pdf
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to read PDF file: {str(e)}"
            )
   
    elif filename.lower().endswith(".docx"):
        try:
            doc = Document(io.BytesIO(content))
            return "\n".join(
                paragraph.text for paragraph in doc.paragraphs
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to read DOCX file: {str(e)}"
            )
    raise HTTPException(
        status_code=415,
        detail="Only PDF and DOCX files are allowed"
    )
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")
    # Validate file type
    if not filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=415,
            detail="Only PDF and DOCX files are allowed"
        )
    
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )
    
    text = extract_text(filename, content)
    # Split text into chunks
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text found in the document"
        )
  
    document_id = str(uuid.uuid4())
  
    try:
        collection.add(
            ids=[
                f"{document_id}_{i}"
                for i in range(len(chunks))
            ],
            documents=chunks,
            metadatas=[
                {
                    "document_id": document_id,
                    "file_name": filename,
                    "chunk_number": i
                }
                for i in range(len(chunks))
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store document in ChromaDB: {str(e)}"
        )
    
    document_info = {
        "id": document_id,
        "name": filename
    }
    documents.append(document_info)
    try:
        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=4)
    except OSError as e:
        documents.pop()
       
        try:
            collection.delete(
                where={"document_id": document_id}
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save document metadata: {str(e)}"
        )
    return JSONResponse(
        status_code=201,
        content={
            "id": document_id,
            "filename": filename,
            "chunks": len(chunks),
            "message": "Document uploaded successfully"
        }
    )
@app.get("/documents")
def list_documents():
    return documents
@app.get("/")
def home():
    return {
        "message": "Document ingestion API is running",
        "status": "success"
    }
import os
from collections import defaultdict

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException


# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=api_key) if api_key else None


# Request validation
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


# Response validation
class Source(BaseModel):
    file: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# Store recent messages per session
chat_memory = defaultdict(list)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()
    session_id = request.session_id.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty"
        )

    if not session_id:
        raise HTTPException(
            status_code=422,
            detail="Session ID cannot be empty"
        )

    if llm is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing. Add it to your .env file."
        )

    history = chat_memory[session_id]

    # Include recent user messages for follow-up questions
    recent_history = history[-6:]

    search_query = " ".join(
        [item["content"] for item in recent_history]
        + [message]
    )

    # Retrieve relevant document chunks
    try:
        results = collection.query(
            query_texts=[search_query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(e)}"
        )

    chunks = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    # Keep sufficiently relevant results
    relevant = [
        (chunk, metadata or {}, distance)
        for chunk, metadata, distance in zip(
            chunks, metadatas, distances
        )
        if distance is not None and distance < 1.4
    ]

    if not relevant:
        answer = "I don't know"
        sources = []

    else:
        context_parts = []
        source_set = set()

        for chunk, metadata, _ in relevant:
            filename = metadata.get("file_name", "Unknown file")
            page = int(metadata.get("page", 1))

            context_parts.append(
                f"Source: {filename}, Page: {page}\n{chunk}"
            )

            source_set.add((filename, page))

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a document question-answering assistant.

Rules:
Answer using only the supplied document context.
Use conversation history only to understand follow-up questions.
Do not treat conversation history as factual evidence.
If the document context does not support the answer, say "I don't know".
Do not invent facts.

Recent conversation:
{recent_history}

Document context:
{context}

Question:
{message}

Answer:
"""

        try:
            response = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            answer = (
                response.choices[0].message.content or "I don't know"
            ).strip()

        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI request failed: {str(e)}"
            )

        sources = [
            {"file": filename, "page": page}
            for filename, page in sorted(source_set)
        ]

    # Save conversation and retain the last 5 messages
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ])

    chat_memory[session_id] = history[-5:]

    return ChatResponse(
        answer=answer,
        sources=[
            Source(**source) for source in sources
        ]
    )