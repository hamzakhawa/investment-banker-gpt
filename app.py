import os
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

from apikey import apikey
os.environ['OPENAI_API_KEY'] = apikey

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('MetaEarnings2024Q1.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='MetaEarnings2024Q1')

vectorstore_info = VectorStoreInfo(
    name = "MetaEarnings2024Q1",
    description="company's earnings report as a pdf",
    vectorstore = store
)

toolkit  = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

agent_executor = create_vectorstore_agent(
    llm = llm,
    toolkit=toolkit,
    verbose=True
)

st.title('💵 Augvestment - A tool to enhance your invesment process')
prompt = st.text_input("Ask about Meta's 2024 Q1 Earnings..." )

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

with st.expander('Document Similarity Search'):
    search = store.similarity_search_with_score(prompt)
    st.write(search[0][0].page_content)

