from fastapi import FastAPI, HTTPException
import google.generativeai as genai

# Configure API key for Google Generative AI
API_KEY = "AIzaSyDHLQe1XH7ZtwwvLrTc3x4Kk5dosQUUmio"
genai.configure(api_key=API_KEY)

insurance_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=genai.GenerationConfig(temperature=0.1))

app = FastAPI()

from fastapi import Request

@app.post("/insurance-chatbot")
async def insurance_chatbot(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        details = """You are an expert insurance advisor and chatbot that 
                    provides detailed and accurate information about various insurance policies. 
                    Your role is to assist users with any and all questions they have about insurance,
                    including but not limited to: types of insurance (health, life, auto, home, etc.),
                    policy coverage details, premium calculations, claim processes, benefits, exclusions,
                    legal terms, and how to choose the right insurance plan. You respond in a clear, concise,
                    and friendly manner, making complex concepts easy to understand for the user. You also provide 
                    real-life examples, industry insights, and step-by-step guidance when necessary.
                    """
        response = insurance_model.generate_content(f"{details}. User Query: {query}")
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Chatbot API is up and running!"}



