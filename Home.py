import streamlit as st

st.set_page_config(
    page_title="Welcome to RAGwise",
    page_icon="👋",
    layout="wide"
)

st.title("Welcome to RAGwise!")

st.markdown("""
Designed to help you interact with documents intelligently.

### 🗿 Getting Started

1. Head over to the **RAGwise** page (👈 select from the sidebar)
2. Upload your documents and/or paste a URL
3. Click "Upload and store data to RAG"
4. Start asking questions and comparing the answers from ChatGPT 3.5
   

### 📑 Supported File Types
  - 📄 PDF Documents
  - 📝 Text Files
  - 🖼️ Images (JPG, JPEG, PNG)
  - 🌐 Web Pages (Just paste the URL)


""")




st.markdown("Built with ❤️ by [Khushi](https://www.linkedin.com/in/khushi-makhecha/) <3")