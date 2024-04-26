import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Markdown as md
from langchain_community.document_loaders import PyPDFLoader

f = open("key.txt")
api_key = f.read()

st.title("RAG PDF Navigator")


user_input = st.text_input("Enter your query")


chat_model = ChatGoogleGenerativeAI(google_api_key=api_key,
                                   model="gemini-1.5-pro-latest",
                                   convert_system_message_to_human=True)

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="I'm always eager to learn and grow, and your feedback is invaluable. Feel free to share any suggestions you have to make me a more effective assistant.  Beyond answering your questions in markdown, I can also craft creative text formats like poems, code, scripts, or even musical pieces. Just give me a starting point and I'll use my knowledge to create something special for you. What task can I tackle for you today? I'm here to support you in any way possible."),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Question: {question}

Answer: """)
])

output_parser = StrOutputParser()

# Loading Document
loader = PyPDFLoader('arxiv.pdf')
pages = loader.load_and_split()

data = loader.load()

# Splitting documents into chunkers
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Creating Chunks Embedding
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key,
                                               model="models/embedding-001")

# Store the chunks in vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a helpful assistant, trained to provide accurate and relevant information based on the context provided.
    Your answers should be formatted in markdown for better readability. """),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Context:
{context}

Question:
{question}

Answer: """)
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

if st.button("Generate"):
    response = rag_chain.invoke(user_input)
    st.write(response)