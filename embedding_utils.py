# import numpy as np
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load_dotenv()

# # Load Google API Key from environment variable
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Create embedding model instance
# EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=GOOGLE_API_KEY
# )

# def embed_texts(texts):
#     """
#     Generate embeddings for a list of text chunks.

#     Args:
#         texts (list): List of text strings.

#     Returns:
#         np.array: Array of embeddings.
#     """
#     embeddings = EMBEDDING_MODEL.embed_documents(texts)
#     return np.array(embeddings)

# def embed_query(query):
#     """
#     Generate embedding for query string.

#     Args:
#         query (str): User's question/query.

#     Returns:
#         np.array: Embedding of query.
#     """
#     embedding = EMBEDDING_MODEL.embed_query(query)
#     return np.array([embedding])

import numpy as np
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

def embed_texts(texts):
    embeddings = EMBEDDING_MODEL.embed_documents(texts)
    return embeddings  # no need to convert to numpy array

def embed_query(query):
    embedding = EMBEDDING_MODEL.embed_query(query)
    return embedding
