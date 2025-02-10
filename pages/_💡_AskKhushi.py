import streamlit as st
from helper.rag_functions import *

st.title("AskKhushi")

st.divider()

if st.button("Test content chunking"):
    full_document = read_pdf_from_directory("pdf")
    chunked_document = chunk_text_for_list(docs=full_document)
    chunked_document_embeddings = generate_embeddings(documents=chunked_document)
    st.success(len(chunked_document_embeddings))
    st.success(chunked_document_embeddings)
else:
    st.warning("Please click on the button to test chunking")