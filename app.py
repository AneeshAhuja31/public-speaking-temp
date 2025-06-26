from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import time
import os
from key_manager import APIKeyManager
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API")

base_prompt = """
You are a speech specialist, based on the text input you will recieve, assess the speech based on grammar and other public speaking skill such that the user can
learn how to improve there speech.
Keep it short, only 2-3 lines.
Here is the speech:
{text}
"""

report_prompt = """
You are a comprehensive speech and presentation coach. Based on the following data from a speech session, provide a detailed analysis and recommendations:

Session Duration: {duration} seconds
Audio Transcripts: {transcripts}
Posture Analysis: Good posture maintained for {good_posture_seconds} out of {total_seconds} seconds ({posture_percentage}%)
Hand Gestures: Hand gestures detected for {hand_gestures_seconds} out of {total_seconds} seconds ({gestures_percentage}%)
Speaking Activity: Active speaking detected for {speaking_seconds} out of {total_seconds} seconds ({speaking_percentage}%)

Please provide:
1. Overall Performance Summary (2-3 sentences)
2. Strengths identified
3. Areas for improvement
4. Specific recommendations for better public speaking
5. Score out of 10 for overall presentation skills

Keep the analysis comprehensive but concise.
"""

prompt_template = ChatPromptTemplate.from_template(base_prompt)
report_template = ChatPromptTemplate.from_template(report_prompt)

key_manager = APIKeyManager()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

class MediaPipeData(BaseModel):
    session_duration: float
    good_posture_seconds: float
    hand_gestures_seconds: float
    speaking_seconds: float

class TextChunk(BaseModel):
    text: str
    response: str
    timestamp: float

class SessionData(BaseModel):
    mediapipe_data: MediaPipeData
    text_chunks: list[TextChunk]

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.session_start_times = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    async def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
            
        except ValueError:
            pass
            
    async def send_text(self, text: str, websocket: WebSocket):
        await websocket.send_text(text)
        
manager = ConnectionManager()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=key_manager.get_next_key()
    )

"""
Note (for Yashika and Divyansh): in the .env file put the 3 Gemini Api keys as follows (follow the exact structure):

GEMINI_API_KEY_1 = 
GEMINI_API_KEY_2 = 
GEMINI_API_KEY_3 = 

"""

async def call_gemini(message):
    llm = get_llm()
    chain = prompt_template|llm|StrOutputParser()
    response = await chain.ainvoke({"text":message})
    return response

async def generate_final_report(session_data: SessionData):
    text_chunks = session_data.text_chunks
    mediapipe_data = session_data.mediapipe_data
    
    transcripts = [chunk.text for chunk in text_chunks]
    
    # Calculate percentages
    total_seconds = mediapipe_data.session_duration
    posture_percentage = round((mediapipe_data.good_posture_seconds / total_seconds) * 100, 1)
    gestures_percentage = round((mediapipe_data.hand_gestures_seconds / total_seconds) * 100, 1)
    speaking_percentage = round((mediapipe_data.speaking_seconds / total_seconds) * 100, 1)
    
    llm = get_llm()
    chain = report_template|llm|StrOutputParser()
    
    report = await chain.ainvoke({
        "duration": total_seconds,
        "transcripts": " ".join(transcripts),
        "good_posture_seconds": mediapipe_data.good_posture_seconds,
        "total_seconds": total_seconds,
        "posture_percentage": posture_percentage,
        "hand_gestures_seconds": mediapipe_data.hand_gestures_seconds,
        "gestures_percentage": gestures_percentage,
        "speaking_seconds": mediapipe_data.speaking_seconds,
        "speaking_percentage": speaking_percentage
    })
    
    return report


@app.post("/submit-session-data")
async def submit_session_data(request: Request):
    try:
        data = await request.json()
        session_data = SessionData(**data)
        
        # Generate final report
        report = await generate_final_report(session_data)
        
        return JSONResponse({
            "status": "success",
            "report": report,
            "session_summary": {
                "duration": session_data.mediapipe_data.session_duration,
                "posture_score": round((session_data.mediapipe_data.good_posture_seconds / session_data.mediapipe_data.session_duration) * 100, 1),
                "gesture_score": round((session_data.mediapipe_data.hand_gestures_seconds / session_data.mediapipe_data.session_duration) * 100, 1),
                "speaking_score": round((session_data.mediapipe_data.speaking_seconds / session_data.mediapipe_data.session_duration) * 100, 1),
                "total_speech_chunks": len(session_data.text_chunks)
            }
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                print(type(data))
                print(f"Received text: {data}")
                
                res = await call_gemini(data)
                    
                print(res)
                await manager.send_text(res, websocket)
                    
            except WebSocketDisconnect:
                await manager.disconnect(websocket)
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                break
    finally:
        await manager.disconnect(websocket)