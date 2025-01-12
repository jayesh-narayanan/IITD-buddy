import streamlit as st
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain



groq_api_key = st.secrets['GROQ_API_KEY']
qdrant_api_key = st.secrets['QDRANT_API_KEY']
qdrant_url = st.secrets['QDRANT_URL']

# Initialize the LLM model (Groq)
llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

# Initialize Qdrant client
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

encoder = SentenceTransformer("all-MiniLM-L6-v2")


# Create a retriever for Qdrant (default: top 5 similar results)
def custom_retriever(query, collection_name, top_k=5):
    query_vector = encoder.encode(query)
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    context = [hit.payload['description'] for hit in search_results]
    return context


# Define the prompt template for the assistant
prompt_template = """
You are an intelligent assistant designed to help IIT Delhi students by answering their queries based on the provided context.
Keep your response concise, between 100 to 200 words, ensuring it is relevant, clear, and easy to understand. If the context is a list of lists, each list in the context is meant to contain words. Only provide the link if it is present in the context otherwise do not provide any links.
Use the following context to respond to the user's question.

Context:
{context}

Question:
{query}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Combine retriever, prompt, and LLM into a chain
chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit app setup
st.title("IITD_buddy")
st.write("Ask any question to your buddy!")

# Input from user
user_query = st.text_input("Enter your question here:",
                           value="")

if st.button("Get Response") or (user_query and user_query.strip() != ""):
    with st.spinner("Generating response..."):
        try:
            # Retrieve context from Qdrant
            context = custom_retriever(user_query, collection_name="my_books")

            # Invoke the chain with both context and query
            response = chain.run({"context": context, "query": user_query})

            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
