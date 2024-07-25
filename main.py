from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llm.llm_util import access_engine 
from fastapi import FastAPI
from fastapi import APIRouter 
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware

from typing import Optional  
from schemas.models import ChatModelSchema 

import dotenv
import os 
import spacy 
import uvicorn 

# loading environment variables 
dotenv.load_dotenv() 

# backend modules 
app = FastAPI() 
router = APIRouter() 

origins = [
        "https://localhost:8000",
] 

# loading the spacy nlp module 
nlp = spacy.load("en_core_web_lg") 

# getting environment variables 
MISTRAL_API_KEY: Optional[str] = os.environ.get("MISTRAL_API_KEY") 
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") 

# defining constants 
BASE_PATH : str = os.getcwd()
DATABASE_PATH : str = os.path.join(BASE_PATH, "dronedb")
CONTENT_PATH : str = os.path.join(BASE_PATH, "drone_content")
COLLECTION : str = "drones"
COURSE_LINK: str = "https://idta.in" 

# defining mistral model 
llm = MistralAI(api_key=MISTRAL_API_KEY)
Settings.llm = llm

# defining the embedding for vectorstore calls 
embed_model = HuggingFaceEmbedding()
Settings.embed_model = embed_model

# getting access to the query engine
query_engine = access_engine(
                content_path=CONTENT_PATH, 
                database_path=DATABASE_PATH, 
                collection_name=COLLECTION, 
                nlp=nlp, 
) 

dotenv.load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# to create a generator that will send tokens to the endpoint 
async def get_response(message: str):

    # defining a prompt for the guidance 
    prompt = ("[INST] You are a drone ai assistant responsible." +
              "Your goal is to answer the following user query: %s[/INST]")

    response = query_engine.query(prompt % message) 
    for token in response.response_gen: 
        yield token 

@app.post("/generate/")
async def generate(chat: ChatModelSchema): 
    return StreamingResponse(
            get_response(chat.message), 
    ) 

if __name__ == "__main__":
    uvicorn.run(app)

