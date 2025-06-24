import PyPDF2

def load_pdf(file):
    """
    Extract text from uploaded PDF file.

    Args:
        file (file-like): Uploaded PDF file.

    Returns:
        str: Extracted text from all pages.
    """
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split large text into smaller overlapping chunks for better embedding.

    Args:
        text (str): Entire document text.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks
