import openai
from flask import Flask, request, render_template, jsonify
import json
from pdfminer.high_level import extract_text
import os
import glob
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key



def get_pdf_text(folder):
  text = ""
  for pdf in os.listdir(folder):
    pdf_reader = PdfReader(os.path.join(folder, pdf))
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text


def get_chunks(text):
  text_splitter = CharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len,
    separator=" \n"
  )
  chunks=text_splitter.split_text(text)
  return chunks


def get_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_texts(texts= text_chunks, embedding=embeddings)
  return vectorstore


def get_conversation_chain(vectorStore):
  llm = ChatOpenAI(temperature=0.3)
  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  conversationChain= ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorStore.as_retriever(),
    memory = memory
  )
  return conversationChain



def delete_files_in_directory(folder_name):
    # Get the current working directory
    directory = os.getcwd()

    # Create the folder path
    folder_path = os.path.join(directory, folder_name)

    # Get a list of all .pdf files in the specified folder
    files = glob.glob(os.path.join(folder_path, "*.pdf"))

    # Loop through each file in the list
    for file in files:
        # Delete the file
        try:
            os.remove(file)
            print(f"File {file} has been deleted successfully")
        except Exception as e:
            print(f"Error occurred while deleting file {file}. Error message: {str(e)}")


# Set up Flask application
app = Flask(__name__)


@app.route('/')
def home():
    delete_files_in_directory('pdfs')
    return render_template('./index.html')


# Set up route to receive WhatsApp messages
@app.route('/get-response', methods=['POST'])
def get_response():

    try:
        global f
        data = request.get_json()

        prompt = data['prompt']
        name = f.filename
        print(name[-3:])

        if (name[-3:] == "pdf"):
            text = get_pdf_text('pdfs')
            chunks = get_chunks(text)
            vectorStore = get_vectorstore(chunks)
            conversation = get_conversation_chain(vectorStore)
            response = conversation({'question': prompt})
            return(response['answer']) 

        else:
            return ("Please upload a pdf file.")
        
    except ValueError:
        return ValueError


@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    delete_files_in_directory('pdfs')
    print("Got request in static files")
    print(request.files)
    global f
    f = request.files['static_file']
    if (f.filename[-3:] == "pdf"):
        file_path = os.path.join('pdfs', f.filename)
        f.save(file_path)
        resp = {"success": True, "response": "File saved!"}
        return jsonify(resp), 200    
    else:
        resp = {"success": False, "response": "Please upload a pdf file."}
        return jsonify(resp), 404


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
