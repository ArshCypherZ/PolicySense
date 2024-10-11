import os
from dotenv import load_dotenv
from fastapi import FastAPI
import google.generativeai as genai
from insurance_bot import router as insurance_bot_router
from autoform_bot import router as autoform_bot_router
from doc_upload_bot import router as doc_upload_bot_router

# Load the .env file
load_dotenv()

# Configure API key for Google Generative AI
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

app = FastAPI()

app.include_router(insurance_bot_router, prefix="/insurance-chatbot")
app.include_router(autoform_bot_router,prefix="/update-form")
app.include_router(doc_upload_bot_router)


# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Chatbot API is up and running!"}

