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
        
        details = """You are an expert insurance advisor and chatbot that provides detailed and accurate information..."""
        response = insurance_model.generate_content(f"{details}. User Query: {query}")
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Chatbot API is up and running!"}

