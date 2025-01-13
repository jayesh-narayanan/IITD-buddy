#CODE FOR PREPROCESSING THE DATA FROM THE PDF'S
import os
import re
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def find_pdfs_in_folder(folder_path):
    pdf_files = []

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isdir(item_path):
            pdf_files.extend(find_pdfs_in_folder(item_path))
        elif item.lower().endswith(".pdf"):
            pdf_files.append((item, item_path))

    return pdf_files


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text


def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def preprocess_data(text_data):
    processed_data = []
    for text in text_data:
        cleaned_text = clean_text(text)
        tokens = tokenize_and_lemmatize(cleaned_text)

        important_words = tokens[:7]

        important_words_text = ' '.join(important_words)

        while len(important_words_text) < 150 and len(tokens) > len(important_words):
            important_words.append(tokens[len(important_words)])
            important_words_text = ' '.join(important_words)

        important_words_text = important_words_text[:250]

        processed_data.append({
            important_words_text: important_words_text
        })

    return processed_data


reference_folder_path = r"C:\Users\shash\Downloads\branches_zipped\branches"

pdf_files = find_pdfs_in_folder(reference_folder_path)

pdf_texts = []
for pdf_file, pdf_path in pdf_files:
    text = extract_text_from_pdf(pdf_path)
    pdf_texts.append(text)

preprocessed_documents = preprocess_data(pdf_texts)

branches = []
for doc in preprocessed_documents:
    branches.append(doc)

print(branches)
