import streamlit as st
import fitz
import io
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

###########################################################################################
# Function to Read the API 
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key
###########################################################################################

# Function to Display the Document  and Formating the Document using PyMuPDF 
# Refer this :
### https://pymupdf.readthedocs.io/en/latest/intro.html

def display_document_preview(pdf_file):
    pdf_document = fitz.open(stream=io.BytesIO(pdf_file.read()), filetype="pdf")
    st.sidebar.markdown("### Document Pages")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_bytes = page.get_pixmap().tobytes("png")
        pil_image = Image.open(io.BytesIO(image_bytes))
        st.sidebar.image(pil_image, caption=f"Page {page_num + 1}", use_column_width=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
##############################################################################################

# UI Development

st.sidebar.title("Document Preview")
uploaded_file = st.sidebar.file_uploader("Upload document file", type=["pdf", "docx"])

if uploaded_file:
    display_document_preview(uploaded_file)

st.title("RAG Document Analyser🥸🔎✨")

# reading the API file 

api_key_file = "Your API Path"
api_key = read_api_key(api_key_file)

# Set up Your path of the Document 

loader = PyPDFLoader("Resume.pdf")
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")

# Applying Chunking for Optimization

data = loader.load_and_split()
text_splitter = NLTKTextSplitter(chunk_size=200, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

# Utilizing the ChromaDataBase for Storing the Document Text data in Vector Format

db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()

# Connect to the persisted Chroma database
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 1})

# Giving the Instructions to the Model

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from the user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

# Instantiate chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-pro-latest")

# Instantiate output parser
output_parser = StrOutputParser()

########################## LangChain Creation ###################################################
# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)
################################################################################################
user_input = st.text_input("Ask a question:")
if st.button("Search"):
    retrieved_docs = rag_chain.invoke(user_input)
    st.write(retrieved_docs)
