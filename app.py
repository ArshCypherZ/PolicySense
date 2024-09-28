from fastapi import FastAPI, HTTPException, UploadFile, File,Form
from pydantic import BaseModel
import google.generativeai as genai
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Configure API key for Google Generative AI
API_KEY = "AIzaSyDHLQe1XH7ZtwwvLrTc3x4Kk5dosQUUmio"
genai.configure(api_key=API_KEY)

# Initialize the Generative AI models
insurance_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=genai.GenerationConfig(temperature=0.1))
form_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=genai.GenerationConfig(temperature=0.1))
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize FastAPI app
app = FastAPI()

# Request model for chatbot queries
class QueryRequest(BaseModel):
    query: str

# Insurance chatbot endpoint
@app.post("/insurance-chatbot")
def insurance_chatbot(request: QueryRequest):
    details = """You are an expert insurance advisor and chatbot that provides detailed and accurate information about various insurance policies. 
    Your role is to assist users with any and all questions they have about insurance, including but not limited to: types of insurance (health, 
    life, auto, home, etc.), policy coverage details, premium calculations, claim processes, benefits, exclusions, legal terms, and how to choose 
    the right insurance plan. You respond in a clear, concise, and friendly manner, making complex concepts easy to understand for the user. 
    You also provide real-life examples, industry insights, and step-by-step guidance when necessary."""
    
    try:
        # Generate a response from the insurance model
        response = insurance_model.generate_content(f"{details}. User Query: {request.query}")
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Document-based chatbot endpoint
@app.post("/document-chatbot")
async def document_chatbot(file: UploadFile = File(...), query: str = Form(...)):
    def extract_text_from_pdf(file_obj):
        text = ""
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                # Extract text from left and right columns if applicable
                left_text = page.within_bbox((0, 0, page.width / 2, page.height)).extract_text() or ""
                right_text = page.within_bbox((page.width / 2, 0, page.width, page.height)).extract_text() or ""
                if left_text and right_text:
                    text += left_text + "\n" + right_text + "\n"
                else:
                    text += page.extract_text() + "\n"
        return text

    def split_text_into_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]

    def find_relevant_sentences(query, sentences):
        sentence_embeddings = text_model.encode(sentences)
        query_embedding = text_model.encode(query)
        distances = util.pytorch_cos_sim(query_embedding, sentence_embeddings)
        top_indices = distances[0].argsort(descending=True)[:3]
        return [sentences[idx] for idx in top_indices]

    try:
        # Directly use the uploaded file without saving to local storage
        pdf_text = extract_text_from_pdf(file.file)
        sentences = split_text_into_sentences(pdf_text)
        relevant_sentences = find_relevant_sentences(query, sentences)

        return {"response": relevant_sentences}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")








# # Form-filling chatbot endpoint
# @app.post("/form-chatbot")
# def form_chatbot(request: QueryRequest):
#     details = """Extract structured information from the following text:

#     Text: {}

#     Provide the output in the following format:

#     1. Name = <name>
#     2. Address = <address>
#     3. Age = <age>"""

#     try:
#         # Generate a response from the form model
#         response = form_model.generate_content(details.format(request.query))
#         return {"response": response.text}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Chatbot API is up and running!"}

