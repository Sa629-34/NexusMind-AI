from fastapi import FastAPI, UploadFile, File
import os
import shutil
from pdf_reader import extract_text
from text_chunker import chunk_text
from vector_store import store_chunks, search_chunks, reset_database
from llm import ask_llama
from pydantic import BaseModel

app = FastAPI(
    title="NexusMind AI",
    version="1.0.0",
    description="Adaptive & Trustworthy Generative AI using RAG",
)

UPLOAD_FOLDER = "../documents"


class Question(BaseModel):
    question: str


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/", summary="Home")
def home():
    return {
        "message": "Welcome to NexusMind AI 🚀"
    }


@app.post(
    "/upload",
    summary="Upload PDF",
    description="Upload a PDF document into NexusMind AI."
)
async def upload_pdf(file: UploadFile = File(...)):

    try:

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Reset old database
        reset_database()

        # Read PDF
        pages = extract_text(file_path)

        # Create chunks
        chunks = chunk_text(pages)

        # Store in ChromaDB
        stored = store_chunks(chunks, file.filename)

        return {
            "message": "PDF Uploaded Successfully",
            "filename": file.filename,
            "total_chunks": len(chunks),
            "stored_chunks": stored
        }

    except Exception as e:

        return {
            "error": str(e)
        }


@app.post(
    "/chat",
    summary="Ask Question",
    description="Ask questions from uploaded PDF using RAG."
)
def chat(data: Question):

    try:

        chunks, metadata, confidence = search_chunks(data.question)

        # Hallucination Detection
        if len(chunks) == 0 or confidence < 30:
            return {
                "question": data.question,
                "answer": "I couldn't find this information in the uploaded document.",
                "confidence": f"{confidence}%",
                "sources": metadata
            }

        # Build Context
        context = "\n\n".join(chunks)

        # Ask Llama
        answer = ask_llama(context, data.question)

        return {
            "question": data.question,
            "answer": answer,
            "confidence": f"{confidence}%",
            "sources": metadata
        }

    except Exception as e:

        return {
            "error": str(e)
        }