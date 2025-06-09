


def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    end = 0

    while end < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start = end - overlap  # Overlap chunks

    return chunks