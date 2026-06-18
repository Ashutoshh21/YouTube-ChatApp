import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1.a DataLoader (Robust YouTube Fetcher)

def manual_ytLoader(id_or_URL: str) -> Document:
    pattern = r'(?:v=|\/shorts\/|\/embed\/|\/vi\/|youtu\.be\/|^\b)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, id_or_URL)
    video_id = match.group(1) if match else id_or_URL

    try:
        # 1. Initializing the modern client instance
        api_client = YouTubeTranscriptApi()
        
        # 2. Extracting as raw dictionary list data
        raw_transcript = api_client.fetch(video_id).to_raw_data()
        
        # 3. Joining dictionary key strings seamlessly using bracket notation
        clean_text = " ".join(doc["text"] for doc in raw_transcript)

        if clean_text.strip():
            return Document(
                page_content=clean_text, 
                metadata={"source": f"https://youtube.com/watch?v={video_id}", "video_id": video_id}
            )
            
    except Exception as e:
        # Throwing an explicit RuntimeError prevents app from trying to run embeddings on non-existent text
        raise RuntimeError(f"File Transcript cannot be loaded: {e}")

    raise RuntimeError("The transcript was fetched but contained no readable text content.")



# 1.b Text Splitting & Vector Storage

def create_retriever(doc: Document):
    
    if not doc:
        raise ValueError("Cannot create a retriever from an empty or missing Document.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunked_docs = splitter.split_documents([doc])

    # Initializing embd model
    embd = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    vector_Store = FAISS.from_documents(
        documents=chunked_docs,
        embedding=embd
    )

    return vector_Store.as_retriever(search_kwargs={"k": 3})


# 2. Context Merger Utilities

def contextMerger(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# 3. Execution Chain

def generate_ans(query: str, retriever) -> str:
    parser = StrOutputParser()
    model = ChatGroq(model="llama-3.3-70b-versatile")
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Help me answer the query: {query}
        Strictly use ONLY reference to the given YouTube transcript context: {context}
        
        If the context doesn't provide sufficient or relevant information, simply say 'I don't know' 
        or mention that the video does not talk about the given query.
        """
    )

    parall = RunnableParallel({
        "query": RunnablePassthrough(),
        "context": retriever | RunnableLambda(contextMerger)
    })

    forward_chain = prompt | model | parser
    final_chain = parall | forward_chain

    return final_chain.invoke(query)
