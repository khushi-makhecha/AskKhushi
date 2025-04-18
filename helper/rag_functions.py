from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import hashlib
from pinecone import Pinecone
from openai import OpenAI
from io import StringIO
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
import requests
from helper.image_processor import process_image


OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
EMBEDDINGS = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]


def read_pdf_from_directory(directory: str) -> list[str]:
    # Initialize a PyPDFDirectoryLoader object with the given directory
    file_loader = PyPDFDirectoryLoader(directory)

    # Load PDF documents from the directory
    documents = file_loader.load()

    # Extract only the page content from each document
    page_contents = [doc.page_content for doc in documents]

    return page_contents


def chunk_text_for_list(docs, max_chunk_size = 1000):
    def chunk_text(text_list, max_chunk_size):
        all_chunks = []
        for text in text_list:
            # Ensure each text ends with a double newline to correctly split paragraphs
            if not text.endswith("\n\n"):
                text += "\n\n"
            # Split text into paragraphs
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""
            # Iterate over paragraphs and assemble chunks
            for paragraph in paragraphs:
                # Check if adding the current paragraph exceeds the maximum chunk size
                if (
                    len(current_chunk) + len(paragraph) + 2 > max_chunk_size
                    and current_chunk
                ):
                    # If so, add the current chunk to the list and start a new chunk
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Add the current paragraph to the current chunk
                current_chunk += paragraph.strip() + "\n\n"
            # Add any remaining text as the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                all_chunks.extend(chunks)
        return all_chunks

    # Apply the chunk_text function to each document in the list
    return [chunk_text(doc, max_chunk_size) for doc in docs]


def generate_embeddings(documents: list[any]) -> list[list[float]]:
    embedded = [EMBEDDINGS.embed_documents(doc) for doc in documents]
    return embedded


def generate_short_id(content: str) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


def map_vector_and_text(
    documents: list[any], doc_embeddings: list[list[float]], username: str = None
) -> list[dict[str, any]]:
    data_with_metadata = []

    for doc_text, embedding in zip(documents, doc_embeddings):
        # Convert doc_text to string if it's not already a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        # Generate a unique ID based on the text content
        doc_id = generate_short_id(doc_text)

        # Create a data item dictionary
        metadata = {"text": doc_text}
        if username:
            metadata["username"] = username

        data_item = {
            "id": doc_id,
            "values": embedding[0],
            "metadata": metadata,  # Include the text and username as metadata
        }

        # Append the data item to the list
        data_with_metadata.append(data_item)

    return data_with_metadata

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(st.secrets["PINECONE_INDEX"])


def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
    try:
        index.upsert(vectors=data_with_metadata)
        return True
    except Exception as e:
        st.write(e)
        return False


def get_query_embeddings(query: str) -> list[float]:
    query_embeddings = EMBEDDINGS.embed_query(query)
    return query_embeddings


def query_pinecone_index(
    query_embeddings: list, top_k: int = 2, include_metadata: bool = True, username: str = None
) -> dict[str, any]:
    filter_dict = None
    if username:
        filter_dict = {"username": {"$eq": username}}
    query_response = index.query(
        vector=query_embeddings,
        top_k=top_k,
        include_metadata=include_metadata,
        filter=filter_dict
    )
    return query_response


def generate_answer(answers: dict[str, any], prompt: str) -> str:
  client = OpenAI(api_key=OPENAI_API_KEY)
  text_content = answers['matches'][0]['metadata']['text']

  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "developer", "content": text_content},
          {
              "role": "user",
              "content": "With the given context provide a better answer to the question: " + prompt,

          }
      ]
  )

  return completion.choices[0].message


def upload_files():
    try:
        uploaded_files = st.file_uploader("Choose files to upload to RAG", accept_multiple_files=True)
        if uploaded_files:
            all_text_content = []
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text_content = ""
                    
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean up the text
                            # Replace multiple spaces with single space
                            page_text = ' '.join(page_text.split())
                            # Split into sentences or natural breaks
                            sentences = page_text.split('.')
                            # Rejoin sentences with proper formatting
                            formatted_text = '. '.join(sentence.strip() for sentence in sentences if sentence.strip())
                            if formatted_text and not formatted_text.endswith('.'):
                                formatted_text += '.'
                            text_content += formatted_text + '\n\n'
                    
                    all_text_content.append(text_content.strip())
                elif uploaded_file.type == "text/plain":
                    # To convert to a string based IO:
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    # To read file as string:
                    string_data = stringio.read()
                    all_text_content.append(string_data)
                elif uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/jpg" or uploaded_file.type == "image/png":
                    caption = process_image(uploaded_file)
                    all_text_content.append(caption)
                else:
                    st.error(f"Unsupported file type: {uploaded_file.type}. Please upload PDF or TXT files only.")
                    return None
                    
            return all_text_content if all_text_content else None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []


def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = {
            'title': soup.title.string,
            'content': soup.get_text()
        }
        return page_content
    except Exception as e:
        st.write(f"Error scraping web page: {str(e)}")
        return ""
