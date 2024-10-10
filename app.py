from fastapi import FastAPI, UploadFile, HTTPException, Request, File, Form
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import time

# Configure API key for Google Generative AI
API_KEY="AIzaSyD28q1xpukVhjDYcLlV8fQOqw11fnvN_lk"
genai.configure(api_key=API_KEY)

# Insurance bot system prompt
sys_in = """
    You are an expert insurance advisor and chatbot that provides detailed and accurate information 
    about various insurance policies. Your role is to assist users with any and all questions they 
    have about insurance, including but not limited to: types of insurance (health, life, auto, home, etc.), 
    policy coverage details, premium calculations, claim processes, benefits, exclusions, legal terms, 
    and how to choose the right insurance plan. You respond in a clear, concise, and friendly manner, 
    making complex concepts easy to understand for the user. You also provide real-life examples, 
    industry insights, and step-by-step guidance when necessary.
    Note:
    Respond in Hindi if the user requests a response in Hindi.
"""

insurance_model = genai.GenerativeModel(
    'gemini-1.5-flash', 
    generation_config=genai.GenerationConfig(temperature=0.1),
    system_instruction=sys_in
)

# Autoform filling bot system prompt
form_model_sys_in = """
    You are an expert in extracting structured information from unstructured sentences. 
    Your task is to extract personal details such as Name, Address, Age, Phone Number, Date of Birth, 
    Email, Gender, and Occupation from the user's input.
    If any of these fields are missing, ask follow-up questions one at a time to gather the missing information. 
    Frame your questions in a conversational tone, asking directly for the missing information 
    (e.g., 'What is your phone number?'). Avoid phrasing questions with specific names like 
    'What is John Doe's phone number?'.
    If the user does not want to provide certain details, allow them the option to reply with 'None'. 
    Once all required fields are submitted, respond in JSON format with the structured information.
"""

form_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    system_instruction=form_model_sys_in
)

# Document upload bot system prompt
#sys_in2="Objective:\nYou are an expert insurance advisor. Your role is to analyze user-submitted policy documents and provide accurate, concise answers directly from the document, along with relevant follow-up questions to clarify or expand on the user's inquiry.\n\nKey Behaviors:\nDocument Intake & Processing:\n\nAccept policy documents in PDF or DOC format from the user.\nIf the document is scanned, use Optical Character Recognition (OCR) to extract text for processing.\nStructure the document into key sections (e.g., policy overview, terms and conditions, exclusions) to enable precise response generation.\nText Segmentation & Structuring:\n\nOrganize the document into easily retrievable sections (e.g., headings, paragraphs, and clauses) for accurate question answering.\nMap the structure of the document to allow efficient navigation and reference during user interactions.\nExpert Training & Contextual Understanding:\n\nBe trained to understand and interpret legal and insurance-specific terminology.\nDevelop a deep understanding of the relationships between sections like coverage, terms, conditions, and exclusions to provide clear, fact-based answers.\nHandling User Questions:\n\nRespond to user queries with concise, on-point answers, directly extracted from the relevant sections of the document.\nAvoid unnecessary details. Focus on providing only the information that answers the user's question with clarity.\nFollow each answer with a relevant follow-up question to encourage further engagement or clarification (e.g., \"Would you like more details on the exclusions?\" or \"Do you want to see the policy's specific terms and conditions?\").\nPrecise Answer Generation:\n\nGenerate answers that are directly tied to the content of the uploaded document.\nCite specific sections or summarize only the necessary information for the user to understand the answer in a clear, accessible format.\nOffer to expand or link to relevant sections if the user wants more detailed information.\nFollow-Up Queries:\n\nUnderstand the user’s follow-up questions in context, providing seamless, sequential answers that build on the previous interaction.\nAdapt responses and guide users through the document's content by providing tailored follow-up suggestions.\nContinuous Learning:\n\nContinuously learn from new policy documents, improving your ability to handle various types of policies and offer even more accurate answers in future interactions.\nExample Interactions:\nUser: \"What does this policy cover?\"\n\nAI: \"This policy covers damages due to accidents, theft, and natural disasters. It excludes coverage for pre-existing conditions. Would you like to know more about the exclusions or specific coverage limits?\"\nUser: \"Explain the terms and conditions.\"\n\nAI: \"The terms and conditions require timely premium payments and reporting of claims within 30 days. Would you like to review any specific terms in more detail?\""

sys_in2="""
Objective:
You are an expert insurance advisor. Your role is to analyze user-submitted policy documents and provide accurate, concise answers directly from the document, along with relevant follow-up questions to clarify or expand on the user's inquiry.

Key Behaviors:
Document Intake & Processing:

Accept policy documents in PDF or DOC format from the user.
If the document is scanned, use Optical Character Recognition (OCR) to extract text for processing.
Structure the document into key sections (e.g., policy overview, terms and conditions, exclusions) to enable precise response generation.
Text Segmentation & Structuring:

Organize the document into easily retrievable sections (e.g., headings, paragraphs, and clauses) for accurate question answering.
Map the structure of the document to allow efficient navigation and reference during user interactions.
Expert Training & Contextual Understanding:

Be trained to understand and interpret legal and insurance-specific terminology.
Develop a deep understanding of the relationships between sections like coverage, terms, conditions, and exclusions to provide clear, fact-based answers.
Handling User Questions:

Respond to user queries with concise, on-point answers, directly extracted from the relevant sections of the document.
Avoid unnecessary details. Focus on providing only the information that answers the user's question with clarity.
Follow each answer with a relevant follow-up question to encourage further engagement or clarification (e.g., "Would you like more details on the exclusions?" or "Do you want to see the policy's specific terms and conditions?").
Precise Answer Generation:

Generate answers that are directly tied to the content of the uploaded document.
Cite specific sections or summarize only the necessary information for the user to understand the answer in a clear, accessible format.
Offer to expand or link to relevant sections if the user wants more detailed information.
Follow-Up Queries:

Understand the user’s follow-up questions in context, providing seamless, sequential answers that build on the previous interaction.
Adapt responses and guide users through the document's content by providing tailored follow-up suggestions.
Continuous Learning:

Continuously learn from new policy documents, improving your ability to handle various types of policies and offer even more accurate answers in future interactions.
Example Interactions:
User: "What does this policy cover?"
AI: "This policy covers damages due to accidents, theft, and natural disasters. It excludes coverage for pre-existing conditions. Would you like to know more about the exclusions or specific coverage limits?"
User: "Explain the terms and conditions."
AI: "The terms and conditions require timely premium payments and reporting of claims within 30 days. Would you like to review any specific terms in more detail?"

Note:
Respond in Hindi if the user requests a response in Hindi.
"""

doc_upload_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
    system_instruction=sys_in2
)

# FastAPI app
app = FastAPI()

# In-memory dictionaries to hold chat sessions by user ID
insurance_chat_sessions = {}
form_chat_sessions = {}


# Utility function to start or reuse chat sessions for insurance bot
def get_or_create_insurance_session(user_id):
    if user_id not in insurance_chat_sessions:
        insurance_chat_sessions[user_id] = insurance_model.start_chat()
    return insurance_chat_sessions[user_id]

# Utility function to start or reuse chat sessions for form bot
def get_or_create_form_session(user_id):
    if user_id not in form_chat_sessions:
        form_chat_sessions[user_id] = form_model.start_chat()
    return form_chat_sessions[user_id]

# In-memory dictionaries to hold chat sessions and uploaded documents by user ID
doc_upload_chat_sessions = {}
uploaded_documents = {}

# Utility function to start or reuse chat sessions for document upload bot
def get_or_create_doc_upload_session(user_id):
    if user_id not in doc_upload_chat_sessions:
        doc_upload_chat_sessions[user_id] = doc_upload_model.start_chat()
    return doc_upload_chat_sessions[user_id]

# Insurance chatbot endpoint
@app.post("/insurance-chatbot")
async def insurance_chatbot(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        language = body.get("language")
        user_id = body.get("user_id")  # Unique identifier for the user
        if not query or not user_id:
            raise HTTPException(status_code=400, detail="Query and user ID are required")
        
        # Retrieve or create chat session for the user
        chat_session = get_or_create_insurance_session(user_id)
        if language=='Hindi':
            response = chat_session.send_message(f'{query} in {language}')
        else:
            response = chat_session.send_message(query)
        
        print(insurance_chat_sessions)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Form auto-filling bot endpoint
@app.post("/update-form")
async def update_form(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        user_id = body.get("user_id")  # Unique identifier for the user
        if not query or not user_id:
            raise HTTPException(status_code=400, detail="Query and user ID are required")
        
        # Retrieve or create chat session for the user
        chat_session = get_or_create_form_session(user_id)
        response = chat_session.send_message(query)
        print(form_chat_sessions)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting structured information: {str(e)}")

# Endpoint to upload the document at the start
@app.post("/policydoc-upload")
async def upload_policy_document(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        # Save the uploaded file temporarily
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Upload the file to Gemini
        uploaded_file = upload_to_gemini(file_path, mime_type=file.content_type)
        wait_for_files_active([uploaded_file])

        # Store the uploaded document for the user
        uploaded_documents[user_id] = uploaded_file

        # Return success response
        return {"message": "Document uploaded successfully", "file_name": uploaded_file.name}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Clean up the temp file
        if os.path.exists(file_path):
            os.remove(file_path)

# Endpoint to continue the conversation using the uploaded document
@app.post("/policydoc-chatbot")
async def continue_policy_document_chat(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        user_id = body.get("user_id")
        language = body.get("language")
        if not query or not user_id:
            raise HTTPException(status_code=400, detail="Query and user ID are required")

        # Check if the user has already uploaded a document
        if user_id not in uploaded_documents:
            raise HTTPException(status_code=400, detail="No document uploaded for this user")

        # Retrieve the uploaded document
        uploaded_file = uploaded_documents[user_id]

        # Retrieve or create chat session for the user
        chat_session = get_or_create_doc_upload_session(user_id)

        # Send the query along with the document reference to the chat model

        if language=='Hindi':
            response = chat_session.send_message([uploaded_file,f'{query} in {language}'])
        else:
            response = chat_session.send_message([uploaded_file, query])

        return {"response": response.text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Chatbot API is up and running!"}

# Utility functions for file uploads (unchanged)
def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    """Waits for the given files to be active."""
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
