import streamlit as st
from helper.rag_functions import *
import json

st.set_page_config(page_title="RAGwise")
st.title("RAGwise")

st.divider()

# Initialize session state for scraped content if it doesn't exist
if 'scraped_content' not in st.session_state:
    st.session_state.scraped_content = {"title": "", "content": ""}

merged_documents = []
extracted_from_uploads = upload_files()

url = st.text_input("Paste the URL below.")
if st.button("Go"):
    scraped_output = scrape_url(url)
    st.session_state.scraped_content.update(scraped_output)

if st.button("Upload and store data to RAG"):
    if extracted_from_uploads:
        merged_documents.append(extracted_from_uploads)

    if st.session_state.scraped_content.get('content', '') != "":
        merged_documents.append([st.session_state.scraped_content['content']])
        merged_documents.append([st.session_state.scraped_content['title']])

    # full_document = read_pdf_from_directory("pdf")
    chunked_document = chunk_text_for_list(docs=merged_documents)
    chunked_document_embeddings = generate_embeddings(documents=chunked_document)
    data_with_meta_data = map_vector_and_text(documents=chunked_document, doc_embeddings=chunked_document_embeddings)
    upsert_success = upsert_data_to_pinecone(data_with_metadata=data_with_meta_data)
    if upsert_success:
        st.success("Data uploaded successfully.")
    else:
        st.error("Failed to upload data.")
else:
    st.warning("Please click on the button to generate vectors.")

prompt = st.text_input("Type in your prompt")
if st.button("Query Chatbot"):
    if prompt:
        query_embeddings = get_query_embeddings(query=prompt)
        matching_record = query_pinecone_index(query_embeddings=query_embeddings)
        llm_response = generate_answer(matching_record, prompt)
        st.success(llm_response)
else:
    st.warning("Please click on the button to see the response from LLM.")