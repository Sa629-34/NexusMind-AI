from pypdf import PdfReader

def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for i, page in enumerate(reader.pages):

        page_text = page.extract_text()

        if page_text:
            pages.append({
                "page": i + 1,
                "text": page_text
            })

    return pages