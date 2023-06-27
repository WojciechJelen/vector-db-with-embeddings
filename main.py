import os
from dotenv import load_dotenv
from langchain import OpenAI, VectorDBQA

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

import pinecone

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_DATABASE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)

if __name__ == "__main__":
    loader = TextLoader(file_path="mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=0,
    )

    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=docsearch,
        return_source_documents=True,
    )
    query = "What is the vector database? Give me a brief description in 15 words."
    result = qa({"query": query})
    print(result)
