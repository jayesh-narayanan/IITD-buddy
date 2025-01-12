from qdrant_client import models, QdrantCllient
from collections import Counter
import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from groq import Groq

llm=Groq(api_key=st.secrets["qroq_api_key"])

encoder = SentenceTransformer("all-MiniLM-L6-v2")


def custom_retriever(query, collection_name="my_books", top_k=5):
    """
    Custom retriever for Qdrant.
    
    Args:
        query_vector (list[float]): The vector representation of the query.
        collection_name (str): The name of the Qdrant collection.
        top_k (int): Number of top results to retrieve.
    
    Returns:
        list[dict]: A list of retrieved documents with their scores.
    """
    # search_results = client.search(
    #     collection_name=collection_name,
    #     query_vector=query_vector,
    #     limit=top_k
    # )
    # # Extract the context from the results
    # context = [result.payload for result in search_results]
    hits = client.query_points(
        collection_name=collection_name,
        query_vector=encoder.encode(query).tolist(),
        limit=3,
    ).points

    context = [result.payload for hit in hits]

prompt_template = """
You are an intelligent assistant tasked with answering user queries based on provided context. 
Use the following context to respond to the user's question.

Context:
{context}

Question:
{query}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

chain = (
    {"context": custom_retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("Interactive Chatbot with Qdrant and Groq")
st.write("Ask any question, and the chatbot will respond using context from the vector database!")

user_query = st.text_input("Enter your question here:", value="What qualities did Phileas Fogg display during his journey?")

if st.button("Get Response"):
    with st.spinner("Generating response..."):
        try:
            response = chain.invoke(user_query)
            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

