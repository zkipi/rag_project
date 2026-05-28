import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()

st.set_page_config(
    page_title="Netflix RAG Assistant",
    layout="wide"
)

st.title("Netflix RAG Assistant")

# LOAD DATA

df = pd.read_csv("data/processed/cleaned_netflix.csv")

# Begränsa dataset för snabbare demo
df = df.sample(500, random_state=36)

# CREATE DOCUMENTS

documents = []

for _, row in df.iterrows():

    document = Document(
        page_content=row["combined_text"],
        metadata={
            "title": row["Title"],
            "category": row["Category"],
            "genre": row["Type"],
            "release_date": row["Release_Date"]
        }
    )

    documents.append(document)

# EMBEDDINGS

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# VECTOR DATABASE

ids = [str(i) for i in range(len(documents))]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    ids=ids
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# LLM

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# USER INPUT

question = st.text_input(
    "Ask something about the Netflix dataset"
)

# RAG PIPELINE

if question:

    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in relevant_docs]
    )

    prompt = f"""
    You are answering questions about a Netflix dataset.

    All titles in the context come from the Netflix dataset.

    Use ONLY the context below.

    If multiple titles match the question:
    - list ALL matching titles
    - include release year if available

    Answer clearly using bullet points.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

# SHOW ANSWER

    st.subheader("AI Answer")
    st.write(response.content)

# SHOW RETRIEVED DOCS

    st.subheader("Retrieved Documents")

    for i, doc in enumerate(relevant_docs, start=1):

        st.markdown(f"### Document {i}")

        st.write(doc.page_content)

        st.divider()