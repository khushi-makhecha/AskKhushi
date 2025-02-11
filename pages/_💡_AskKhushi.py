import streamlit as st
from helper.rag_functions import *

st.title("AskKhushi")

st.divider()

if st.button("Test data with meta data"):
    full_document = read_pdf_from_directory("pdf")
    chunked_document = chunk_text_for_list(docs=full_document)
    chunked_document_embeddings = generate_embeddings(documents=chunked_document)
    data_with_meta_data = map_vector_and_text(documents=chunked_document, doc_embeddings=chunked_document_embeddings)
    # Dimension of each vector = 1536
    st.success(len(data_with_meta_data[0]["values"]))
    st.success(data_with_meta_data)
else:
    st.warning("Please click on the button to test vector and text mapping")