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
Keep your response concise, between 100 to 200 words, ensuring it is relevant, clear, and easy to understand. If the context is a list of lists, each list in the context is meant to contain words. If the user asks for resources for exams like Quizzes, Minor, Major, Previous year questions (PYQs) then provide the user with the following link mentioning that this is Yash Agarwal's one drive link : "https://csciitd-my.sharepoint.com/personal/ee1210638_iitd_ac_in/_layouts/15/onedrive.aspx?csf=1&web=1&e=4P9Gee&CID=00e4e4f8%2Dfd7a%2D4ca0%2D9933%2Da92d0a41beaf&id=%2Fpersonal%2Fee1210638%5Fiitd%5Fac%5Fin%2FDocuments%2FFreshie%20Resources%20%28Not%20Updated%29&FolderCTID=0x012000D17E58B562B288428ACE967D9E7A4346&view=0".
Also if the user asks for Faculty Homepage for doing Research projects and any course projects under professors then provide the following link : "https://faculty-homepage.vercel.app/"
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
