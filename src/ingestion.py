import re

from pypdf import PdfReader


def load_document(documents_list):
    text = []

    for doc in documents_list:
        document_path = str(doc)

        reader = PdfReader(document_path)

        for i, page in enumerate(reader.pages):
            text.append({
                "document": document_path,
                "page": i + 1,
                "text": page.extract_text()
            })

    return text


def cleaning(documents_list):
    cleaned_results = []

    for doc in documents_list:
        text = doc["text"]

        if text is None:
            continue

        text = text.strip()

        if text == "":
            continue

        text = re.sub(r"[ \t]+", " ", text)

        # Reduce 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        cleaned_results.append({
            "document": doc["document"],
            "page": doc["page"],
            "text": text
        })

    return cleaned_results