from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

#1.a. DataLoader
#Load the transcript as Document via youtube-transcript-api 
import re
def manual_ytLoader(id_or_URL: str) -> Document:
    # Regex to handle regular URLs, shorts, youtu.be, and raw IDs
    pattern = r'(?:v=|\/shorts\/|\/embed\/|\/vi\/|youtu\.be\/|^\b)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, id_or_URL)
    
    if match:
        video_id = match.group(1)
    else:
        video_id = id_or_URL  # Fallback to the input string
    try:
        # 1. Initialize the API instance
        api_client = YouTubeTranscriptApi()
        
        # 2. Use .fetch() to get the transcript data
        raw_transcript = api_client.fetch(video_id)

        clean_text = " ".join(doc.text for doc in raw_transcript)

        if clean_text.strip():
            return Document(
                page_content=clean_text, 
                metadata={"source": f"<https://youtube.com/watch?v={video_id}>", "video_id": video_id}
            )
    except Exception as e:
        print(f"File Transcript cannot be loaded: {e} \n")


def create_retriever(doc : Document) -> list[Document]:
    #1.b TextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunked_docs = splitter.split_documents([doc])

    #1.c initializing the vector_store

    #first embedding model
    embd = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    vector_Store = FAISS.from_documents(
        documents= chunked_docs,
        embedding=embd
    )

    #2 Retrieval return
    return vector_Store.as_retriever(search_kwargs = {"k":3})



def contextMerger(docs : list[Document]) -> str:
    return " " \
    "\n\n".join(doc.page_content for doc in docs)


def generate_ans(query : str, retriever ) -> str:

    parser = StrOutputParser()
    model = ChatGroq(model = "llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_template(
        """
        you are a helpful assistant. Help me answer the query : {query} , strictly with ONLY reference to
        the given Youtube transcript context : {context},
        If the context doesn't provide sufficient or relevant information simple say 'I don't know' or mention 
        that the video does not talk about the given query.
    """
    )

    parall = RunnableParallel({
        "query" : RunnablePassthrough(),
        "context" : retriever | RunnableLambda(contextMerger)
    })

    forward_chain = prompt | model | parser

    final_chain = parall | forward_chain

    return final_chain.invoke(query)
