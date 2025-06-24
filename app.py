# import os
# import streamlit as st
# from pdf_utils import load_pdf, split_text
# from embedding_utils import embed_texts
# from faiss_utils import (
#     create_faiss_index, save_faiss_index, save_chunks, faiss_index_exists
# )
# from rag_pipeline import search_and_retrieve, ask_gemini

# # Create data directory if not exists
# os.makedirs("data", exist_ok=True)

# # Streamlit page configuration
# st.set_page_config(page_title="Gemini 1.5 Flash Search Genius", page_icon="📄")
# st.title("📄 Search Genius - Document Q&A")

# # File uploader to allow user to upload PDF
# uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# if uploaded_file is not None:
#     # Check if FAISS index already exists
#     if not faiss_index_exists():
#         st.write("Processing uploaded document...")
#         # Load and extract text from PDF
#         text = load_pdf(uploaded_file)
#         # Split extracted text into smaller chunks
#         chunks = split_text(text)
#         # Generate embeddings for each text chunk
#         embeddings = embed_texts(chunks)
#         # Create FAISS index using embeddings
#         index = create_faiss_index(embeddings)
#         # Save the index and chunks locally for reuse
#         save_faiss_index(index)
#         save_chunks(chunks)
#         st.success("Document indexed successfully!")
#     else:
#         st.warning("Index already exists. Delete files in /data folder to re-upload new file.")

# # If FAISS index exists, allow user to ask questions
# if faiss_index_exists():
#     question = st.text_input("Ask a question from the document:")
#     if question:
#         # Retrieve relevant chunks based on question embedding
#         retrieved_chunks = search_and_retrieve(question)
#         # Generate answer using Gemini model
#         answer = ask_gemini(question, retrieved_chunks)
#         st.write("### Answer:")
#         st.write(answer)

#         # Show retrieved chunks for transparency
#         with st.expander("Show retrieved context chunks"):
#             for i, chunk in enumerate(retrieved_chunks):
#                 st.write(f"**Chunk {i+1}:** {chunk}")

import os
import streamlit as st
from pdf_utils import load_pdf, split_text
from embedding_utils import embed_texts
from chroma_utils import (
    get_chroma_collection, save_chunks, chroma_index_exists
)
from rag_pipeline import search_and_retrieve, ask_gemini

os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="Gemini 1.5 Flash Search Genius", page_icon="📄")
st.title("📄 Search Genius - Document Q&A")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    if not chroma_index_exists():
        st.write("Processing uploaded document...")
        text = load_pdf(uploaded_file)
        chunks = split_text(text)
        embeddings = embed_texts(chunks)
        collection = get_chroma_collection()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(i)]
            )
        save_chunks(chunks)
        st.success("Document indexed successfully!")
    else:
        st.warning("Index already exists. Delete files in /data folder to re-upload new file.")

if chroma_index_exists():
    question = st.text_input("Ask a question from the document:")
    if question:
        retrieved_chunks = search_and_retrieve(question)
        answer = ask_gemini(question, retrieved_chunks)
        st.write("### Answer:")
        st.write(answer)

        with st.expander("Show retrieved context chunks"):
            for i, chunk in enumerate(retrieved_chunks):
                st.write(f"**Chunk {i+1}:** {chunk}")
