import chromadb
from sentence_transformers import SentenceTransformer

# Embedding model load
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB Client
client = chromadb.PersistentClient(path="../database")

collection = client.get_or_create_collection("nexusmind")


def store_chunks(chunks, filename):

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        ids.append(str(i))

        documents.append(chunk["text"])

        metadatas.append({
            "pdf": filename,
            "page": chunk["page"],
            "chunk": i + 1
        })

    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(documents)


def search_chunks(query, top_k=3):

    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print("RESULTS =", results)

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    # Agar koi result nahi mila
    if len(distances) == 0:
        return [], [], 0

    # Confidence calculate
    avg_distance = sum(distances) / len(distances)
    confidence = round(100 / (1 + avg_distance))

    # Confidence ko 0-100 ke beech rakho
    confidence = max(0, min(100, confidence))

    return chunks, metadata, confidence


def reset_database():

    global collection

    try:
        client.delete_collection("nexusmind")
    except:
        pass

    collection = client.get_or_create_collection("nexusmind")