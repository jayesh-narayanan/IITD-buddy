#CODE FOR MANAGING (DOING SOME CRUD ACTIONS) THE DATA IN OUR DATABASE OR CLOUD (QDRANT)
from collections import Counter
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient

encoder = SentenceTransformer("all-MiniLM-L6-v2")

file_path_cos = r"C:\Users\shash\Downloads\COS_pre-processed data.txt"
file_path_bsw = r"C:\Users\shash\Downloads\BSW.txt"
file_path_timetable = r"C:\Users\shash\Downloads\timetable_branches.txt"
file_path_branches = r"C:\Users\shash\Downloads\branches.txt"
file_path_cos_start = r"C:\Users\shash\Downloads\cos_start.txt"
file_path_inception = r"C:\Users\shash\Downloads\inception.txt"
file_path_depc_criteria = r"C:\Users\shash\Downloads\depC Criteria.txt"
file_path_extra_info = r"C:\Users\shash\Downloads\Extra_info.txt"

def process_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()

    words = file_content.split()

    documents = []

    chunk_start = 0
    while chunk_start < len(words):
        chunk_size = min(50, len(words) - chunk_start)
        chunk_end = chunk_start + chunk_size
        chunk_words = words[chunk_start:chunk_end]

        word_count = Counter(chunk_words)
        most_frequent_word, _ = word_count.most_common(1)[0]

        document = {
            "name": most_frequent_word,
            "description": " ".join(chunk_words)
        }

        documents.append(document)

        chunk_start = chunk_end

    return documents

cos_documents = process_file(file_path_cos)
bsw_documents = process_file(file_path_bsw)
timetable = process_file(file_path_timetable)
branches_documents = process_file(file_path_branches)
cos_start_documents = process_file(file_path_cos_start)
inception_documents = process_file(file_path_inception)
depc_criteria_documents = process_file(file_path_depc_criteria)
extra_info_documents = process_file(file_path_extra_info)

documents = cos_documents + bsw_documents + timetable + branches_documents + cos_start_documents + inception_documents + depc_criteria_documents + extra_info_documents

# client = QdrantClient(":memory:")  # WE CAN USE THIS LINE OF CODE IF WE NEED TO STORE THE DATA IN OUR LOCAL HOST
api_url = 'https://34d86b00-c62f-447e-bb67-9505781929ed.us-east4-0.gcp.cloud.qdrant.io:6333'
api_key = 'M0XlruJ7tpQpCKqpDJ4sYc_NI66xX2o17j0Wty_nS2_fBrl5YKjvZA'  # Replace with your actual API key

client = QdrantClient(url=api_url, api_key=api_key)
client.delete_collection(collection_name="my_books")


client.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)

client.upload_points(
    collection_name="my_books",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)
