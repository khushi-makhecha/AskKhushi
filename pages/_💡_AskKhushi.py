import streamlit as st
from helper.rag_functions import *

st.title("AskKhushi")

st.divider()

if st.button("Test Pinecone Index"):
    full_document = read_pdf_from_directory("pdf")
    chunked_document = chunk_text_for_list(docs=full_document)
    chunked_document_embeddings = generate_embeddings(documents=chunked_document)
    data_with_meta_data = map_vector_and_text(documents=chunked_document, doc_embeddings=chunked_document_embeddings)
    upsert_success = upsert_data_to_pinecone(data_with_metadata= data_with_meta_data)
    st.success(upsert_success)
else:
    st.warning("Please click on the button to test pinecone records")