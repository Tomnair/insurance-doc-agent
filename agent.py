from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def load_documents(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_knowledge_base(docs):
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return FAISS.from_documents(docs, embeddings)

def create_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template("""
You are an insurance document assistant. 
Answer the question based only on the following context:

{context}

Question: {question}
""")
    chain = (
        {
            "context": vectorstore.as_retriever(),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    print("Insurance Document Q&A Agent")
    print("=" * 40)

    docs = load_documents("sample_policy.txt")
    vectorstore = create_knowledge_base(docs)
    chain = create_chain(vectorstore)

    questions = [
        "What does this policy cover?",
        "What is excluded from coverage?",
        "What is the claims process?"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = chain.invoke(question)
        print(f"A: {answer}")