import streamlit as st
from io import StringIO
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


import os
os.environ["OPENAI_API_KEY"] = ""

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

st.set_page_config(page_title="Document Q&A", page_icon=":smile:")
st.title("Document Q&A")

st.markdown("Paste your document below and ask your question.")

uploaded_file = st.file_uploader("Choose a file", type=["txt"])
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    f = open("doc.txt", "w")
    f.write(string_data)
    f.close()
    loader = UnstructuredFileLoader("./doc.txt")
    docs = loader.load()
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

question = st.text_input(label="",placeholder="Ask your question here")
answer = 0
if st.button("Ask"):
    with st.spinner(text="In progress..."):
        answer = qa.run(question)

if answer:
    ans_area = st.markdown(answer)