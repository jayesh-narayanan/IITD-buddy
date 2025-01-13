IITD Buddy: Intelligent Campus Navigator

Overview:
IITD Buddy is an AI-powered assistant designed to enhance the campus experience for IIT Delhi students. The project aims to simplify campus navigation, provide academic guidance, mentorship, and information on hostel facilities, campus events, and local hotspots. With room for further enhancements, this project seeks to enrich student life by leveraging cutting-edge AI tools.

Problem Statement:
Design an intelligent assistant to:
1)Provide academic and mentorship guidance.
2)Offer information on hostel facilities, campus events, and local hotspots.

Methodology:
Data Collection
Gather data from reliable sources such as:
Courses of Study documents, The BSW Website, Professors’ web pages, Inception(fresher's magazine).

AI Integration:
We have used LLM APIs like Groq for generating intelligent responses.
We have built a vector database with Qdrant for efficient context retrieval.

Workflow:
Cloud setup on Qdrant.
Make a vector database in Qdrant.
Use the RAG (Retrieve and Generate) approach to retreive the top 5 most relevant chunks of text.
Pass the context to the LLM to generate concise and relevant answers.
Display the output of the llm to the user on the app.
Implement a user-friendly interface using Streamlit.

Code Structure:
The pdf's taken to manage the database are preprocessed by using "Preprocessing.py" file
Then we add into our DataBase by using "DBMS.py" file
We make the functioning our bot into a user friendly web and mobile app with the help of Stramlit.

Features:
Academics: Provide guidance on courses, degree requirements, and workload balancing.
Mentorship: Assist students with mentorship opportunities and career planning.
Campus Navigation: Offer directions to departments, hostels, and local hotspots.
Event Updates: Notify students about campus events and activities.

Tools and Technologies:
LLM APIs: Groq / Gemini for intelligent text generation.
Vector Database: Qdrant for efficient data storage and retrieval.
Frontend Framework: Streamlit for building an interactive user interface.
