# web_fallback.py
import os
import json
import time
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from ddgs import DDGS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
JSONL_FILE = "kb.jsonl"
SIMILARITY_THRESHOLD = 0.25  # Further lowered for better FAISS matching
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def extract_urls(query, max_retries=3, timeout=10):
    """Extract URLs from DuckDuckGo search results using DDGS."""
    urls = []
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query=query, max_results=3)
                urls = [result['href'] for result in results if 'href' in result]
            logger.info(f"Extracted URLs for query '{query}': {urls}")
            return urls[:3]
        except Exception as e:
            logger.error(f"Search error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    logger.error(f"Failed to extract URLs for query '{query}' after {max_retries} retries")
    return []

def web_scrape_answer(query, llm):
    """Handle web scraping fallback, store data, and generate answer."""
    # Initialize embeddings and load FAISS
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    
    # Step 1: Search for relevant URLs
    urls = extract_urls(query)
    
    if not urls:
        logger.warning(f"No URLs found for query: {query}")
        return "Sorry, couldn't find relevant web data to scrape."

    # Step 2: Scrape URLs using Playwright
    try:
        loader = PlaywrightURLLoader(
            urls=urls,
            remove_selectors=["header", "footer"],
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        new_docs = loader.load()
        time.sleep(2)  # Avoid rate limits
        logger.info(f"Scraped {len(new_docs)} documents from {len(urls)} URLs")
    except Exception as e:
        logger.error(f"Error during web scraping: {e}")
        return f"Error during web scraping: {str(e)}"

    # Step 3: Process scraped data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(new_docs)
    
    # Step 4: Store in JSONL file with labeled question
    scraped_data = [doc.page_content for doc in new_docs]
    try:
        with open(JSONL_FILE, 'a') as f:
            f.write(json.dumps({"question": query, "data": scraped_data}) + '\n')
        logger.info(f"Saved scraped data to {JSONL_FILE} for query: {query}")
    except Exception as e:
        logger.error(f"Error writing to JSONL: {e}")
    
    # Step 5: Add to FAISS
    try:
        vectorstore.add_documents(splits)
        vectorstore.save_local(DB_FAISS_PATH)
        logger.info("Updated FAISS vector store")
    except Exception as e:
        logger.error(f"Error updating FAISS: {e}")
    
    # Step 6: Generate answer using updated FAISS
    system_prompt = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say you don’t have knowledge about this, don’t try to make up an answer.
    Only use the given context, nothing else.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 3}), qa_chain)
    response = retrieval_chain.invoke({"input": query})
    
    return response["answer"]