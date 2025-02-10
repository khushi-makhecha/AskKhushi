import streamlit as st
from helper.rag_functions import *

st.title("AskKhushi")

st.divider()

if st.button("Test PDF content extraction"):
    full_document = read_pdf_from_directory("pdf")
    st.success(full_document)
else:
    st.warning("Please click on the button to test PDF extraction")