from fastapi import HTTPException, Request, APIRouter
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

router = APIRouter()

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
    Respond in Hindi only if the user requests a response in Hindi.
"""

insurance_model = genai.GenerativeModel(
    'gemini-1.5-flash', 
    generation_config=genai.GenerationConfig(temperature=0.1),
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    system_instruction=sys_in
)

insurance_chat_sessions = {}

# Utility function to start or reuse chat sessions for insurance bot
def get_or_create_insurance_session(user_id):
    if user_id not in insurance_chat_sessions:
        insurance_chat_sessions[user_id] = insurance_model.start_chat()
    return insurance_chat_sessions[user_id]

# Insurance chatbot endpoint
#@app.post("/insurance-chatbot")
@router.post("/")
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
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")