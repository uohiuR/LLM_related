import warnings
warnings.filterwarnings("ignore")
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain,RetrievalQA


data_folder = "data"
file_filter = "*.pdf"

loader = DirectoryLoader(data_folder, file_filter,loader_cls=PyPDFLoader)
data = loader.load()
# Split

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"],
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
all_splits = text_splitter.split_documents(data)


# LLM
# Select the LLM that you downloaded
llm = ChatOllama()

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-private",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()



# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


qa_chain=RetrievalQA.from_chain_type(llm,retriever=retriever,return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})


question="Explain the GCNN architecture used in DEL dataset"
result = qa_chain.invoke({"query": question})
print(result["result"])

for i in range(len(result["source_documents"])):
    print(result["source_documents"][i].metadata)