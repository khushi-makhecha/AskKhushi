import streamlit as st
from helper.rag_functions import *

st.title("AskKhushi")

st.divider()

if st.button("Test Pinecone Index"):
    full_document = read_pdf_from_directory("pdf")
    chunked_document = chunk_text_for_list(docs=full_document)
    chunked_document_embeddings = generate_embeddings(documents=chunked_document)
    data_with_meta_data = map_vector_and_text(documents=chunked_document, doc_embeddings=chunked_document_embeddings)
    # upsert_success = upsert_data_to_pinecone(data_with_metadata= data_with_meta_data)
else:
    st.warning("Please click on the button to test pinecone records")

prompt = st.text_input("Type in your prompt")
if st.button("Test with prompt"):
    if prompt:
        query_embeddings = get_query_embeddings(query=prompt)
        matching_record = query_pinecone_index(query_embeddings=query_embeddings)
        llm_response = generate_answer(matching_record, prompt)
        st.success(llm_response)
else:
    st.warning("Please click on the button to test the LLM response")