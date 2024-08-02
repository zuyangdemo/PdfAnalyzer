from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os


def generate_result(question, memory, openai_api_key, file_docs):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        openai_api_base=os.getenv("OPENAI_API_BASE_V1"),
    )

    file_content = file_docs.read()
    temp_file_path = 'temp.pdf'
    with open(temp_file_path, 'wb') as file:
        file.write(file_content)

    pdf_loader = PyPDFLoader(temp_file_path)
    docs = pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
    )

    texts = splitter.split_documents(docs)

    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key,
        openai_api_base=os.getenv("OPENAI_API_BASE_V1")
    )

    db = FAISS.from_documents(texts, embedder)
    retriever = db.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    response = qa.invoke({"chat_history": memory, "question": question})
    return response
