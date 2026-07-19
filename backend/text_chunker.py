from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    all_chunks = []

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        chunks = splitter.split_text(text)

        for chunk in chunks:

            all_chunks.append({
                "page": page_number,
                "text": chunk
            })

    return all_chunks