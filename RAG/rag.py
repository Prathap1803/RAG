from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Initialize Ollama LLM with dolphin-mistral
llm = OllamaLLM(
    model="dolphin-mistral",
    base_url="http://localhost:11434",
    temperature=0.3  # Lower for factual answers
)

# Initialize local embeddings (no API required)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
def load_documents(file_path):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    pages = loader.load_and_split()
    return pages
        
# Split text into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create vector store
def create_vectorstore(chunks, persist_dir="./chroma_db"):
    if not chunks:
        raise ValueError("No text chunks were created from PDF. Check your loader.")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
        
    )
    return vectorstore

# Load existing vector store
def load_vectorstore(persist_dir="./chroma_db"):
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore

# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve 3 most relevant chunks
    )
    
    template = """
You are a helpful assistant answering questions based on provided documents.
Use only the context provided to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question."

Question: {question}

Context:
{context}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Build the RAG chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Main execution
if __name__ == "__main__":
    # First time setup: Load PDF and create vector store
    if not os.path.exists("./chroma_db"):
        print("Creating vector store from PDF...")
        
        #load files 

        folder_path = "data"
        all_docs = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                documents = load_documents(file_path)
                all_docs.extend(documents)
            except ValueError:
                print(f"skipping unsupported file: {file_name}")

       
        
        # Split into chunks
        chunks = split_documents(all_docs)
        
        # Create vector store
        vectorstore = create_vectorstore(chunks)
        print(f"Vector store created with {len(chunks)} chunks")
    else:
        print("Loading existing vector store...")
        vectorstore = load_vectorstore()
    
    # Create RAG chain
    chain = create_rag_chain(vectorstore)
    
    # Query examples
    
    
    while True:
        query = input("Enter exit/quit to exit: \nPlease eneter your query here:")
        if query.lower() in ["exit", "quit"]:
            print("Exiting Bye....")
            break

        response = chain.invoke(query)
        print(f"Answer: {response}\n")



    