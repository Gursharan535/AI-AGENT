# Medibot.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from web_fallback import web_scrape_answer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"
SIMILARITY_THRESHOLD = 0.25  # Lowered to allow more FAISS matches

# Force PyTorch to use CPU to avoid CUDA warning
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Embeddings & VectorStore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Load Gemini LLM
@st.cache_resource
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5,
        max_tokens=512,
    )
    return llm

# Custom Prompt for Retrieval
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don’t have knowledge about this, don’t try to make up an answer.
Only use the given context, nothing else.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Build QA Chain
@st.cache_resource
def get_qa_chain(llm):
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )
    return qa_chain

# Streamlit App
def main():
    st.title("MediBot - PDF + Web Knowledge Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load LLM and QA chain
    llm = load_llm()
    qa_chain = get_qa_chain(llm)

    # Clear cache button
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

    # Show chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User Input
    user_input = st.chat_input("Ask your question here...")

    if user_input:
        # Display user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Step 1: Get FAISS retrieved documents with similarity scores
        db = get_vectorstore()
        retrieved_docs_with_scores = db.similarity_search_with_score(user_input, k=3)
        retrieved_docs = [doc for doc, score in retrieved_docs_with_scores if score <= SIMILARITY_THRESHOLD]
        
        logger.info(f"Retrieved docs: {len(retrieved_docs)} with scores: {[score for _, score in retrieved_docs_with_scores]}")

        if not retrieved_docs:  # No relevant docs in FAISS
            logger.info("No relevant FAISS docs, falling back to web")
            with st.spinner("Searching the web for you..."):
                answer = web_scrape_answer(user_input, llm)
                source = "Web"
        else:
            # Step 2: Use FAISS QA chain
            response = qa_chain.invoke({"query": user_input})
            answer = response["result"]
            logger.info(f"QA chain answer: {answer}")

            # Step 3: Broader check for no knowledge
            normalized_answer = answer.lower().replace("'", "").replace("’", "")
            no_knowledge_phrases = ["i dont have knowledge", "i dont know", "no information", "unable to answer"]
            if not answer.strip() or any(phrase in normalized_answer for phrase in no_knowledge_phrases):
                logger.info("QA chain failed, falling back to web")
                with st.spinner("Searching the web for you..."):
                    answer = web_scrape_answer(user_input, llm)
                    source = "Web"
            else:
                source = "FAISS"

        # Show assistant reply with source
        st.chat_message("assistant").markdown(f"**[Source: {source}]**\n\n{answer}")
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show FAISS source documents if any
        if retrieved_docs:
            with st.expander("Source Documents"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content[:500] + "...")

if __name__ == "__main__":
    main()