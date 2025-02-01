import os
import re
import json
import random
import time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import speech_recognition as sr
import pyttsx3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

# Load the JSON file for configuration
with open('config.json', 'r') as file:
    config = json.load(file)

# Set up the GROQ API key
os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

# Initialize the recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def preprocess_text(text):
    """
    Cleans and normalizes text for better embeddings.
    - Removes special characters, punctuation, and extra spaces.
    - Converts text to lowercase.
    - Removes stopwords.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_document(document_path):
    """Loads and preprocesses a document for embedding."""
    loader = UnstructuredFileLoader(document_path)
    documents = loader.load()
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
    return documents

def text_to_speech(text):
    """Converts text to speech using pyttsx3."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_for_command():
    """Listens for voice commands and returns them as text."""
    try:
        with sr.Microphone() as source:
            print("Listening for your command...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def integrate_rlhf(prompt, response):
    """Applies RLHF without affecting the response."""
    class SimulatedRewardModel:
        @staticmethod
        def evaluate(prompt, response):
            return len(response) / (len(prompt) + 1)
    
    reward_model = SimulatedRewardModel()
    reward_score = reward_model.evaluate(prompt, response)
    print(f"RLHF Reward Score: {reward_score}")
    return response

def initialize_model():
    """Initializes the QA chain and vector store."""
    global qa_chain, plant_context

    document_path = "plantemissions.pdf"  # Replace with your plant care document path
    documents = preprocess_document(document_path)

    text_splitter = CharacterTextSplitter(
        chunk_size=30000,
        chunk_overlap=400
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    persist_directory = "doc_db"
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        model="llama-3.3-70b-specdec",
        temperature=0.7,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    plant_context = """
    You are a plant care expert named udhyanapalaka with extensive knowledge of gardening, plant health, and environmental factors.
    You provide detailed, accurate, and empathetic advice to plant enthusiasts.
    Respond in a friendly and engaging manner, ensuring your guidance is practical and actionable.
    Give your response in 200 words or less.
    """

def process_query(query):
    """Processes a query using the QA chain and applies RLHF."""
    if not qa_chain:
        print("Model is not initialized.")
        return

    expert_prompt = f"{plant_context}\n\nGardener: {query}\nPlant Expert:"
    response_data = qa_chain.invoke({"query": expert_prompt})
    result_text = response_data["result"]

    optimized_response = integrate_rlhf(expert_prompt, result_text)
    text_to_speech(optimized_response)
    print(f"Plant Expert: {optimized_response}")

if __name__ == "__main__":
    initialize_model()

    while True:
        print("Say 'start' to begin or 'exit' to end the program.")
        command = listen_for_command()
        
        if command == "exit":
            print("Exiting program.")
            break
        elif command == "start":
            while True:
                print("Listening for your query (Say 'exit' to stop)...")
                query = listen_for_command()
                if query == "exit":
                    print("Stopping interaction. Say 'start' to begin again.")
                    break
                elif query:
                    process_query(query)
                else:
                    print("No valid query detected.")
