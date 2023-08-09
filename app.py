import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from PIL import Image
#from dotenv import load_dotenv

#load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pinecone.init(
    api_key=pinecone_api_key,
    environment="asia-southeast1-gcp-free"
)

embeddings = OpenAIEmbeddings()
index_name = "cod5"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

llm = OpenAI(model_kwargs={"api_key": openai_api_key})

qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

left_co, cent_co,last_co = st.columns([1,3,1])
with cent_co:
    image = Image.open('chappyai.png')
    st.image(image, clamp=True, width=400)
    # st.markdown the title

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        model=st.session_state["openai_model"]
        stream=True,
        response = qa_with_sources({"query": prompt})
        full_response += response["result"] + " "
        message_placeholder.markdown(full_response["result"])
        
    st.session_state.messages.append({"role": "assistant", "content": full_response["result"]})
