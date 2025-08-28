# connect_memory_with_llm.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Step 1: Setup LLM (Gemini with Google API)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
print("GOOGLE_API_KEY:", GOOGLE_API_KEY)  # Debug statement to verify token
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5,
        max_tokens=512
    )
    return llm

# Step 2: Connect LLM with FAISS and Create Chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't have knowledge about this, don't try to make up an answer,
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load DATA
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULTS:", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
