from fastapi import HTTPException, Request, APIRouter
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

router = APIRouter()

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

# In-memory dictionaries to hold chat sessions by user ID
form_chat_sessions = {}

# Utility function to start or reuse chat sessions for form bot
def get_or_create_form_session(user_id):
    if user_id not in form_chat_sessions:
        form_chat_sessions[user_id] = form_model.start_chat()
    return form_chat_sessions[user_id]

# Form auto-filling bot endpoint
#@app.post("/update-form")
@router.post("/")
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
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting structured information: {str(e)}")
