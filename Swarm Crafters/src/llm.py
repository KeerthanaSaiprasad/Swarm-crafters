from langchain_google_genai import ChatGoogleGenerativeAI
import os
from callback_handler import GoogleCallbackHandler

class Gemini:
    def __init__(self):
        self.llm = None
        self.llm_cb = None
        self.model_id = os.getenv("model_id")
    def get_llm(self):
        self.llm_cb = GoogleCallbackHandler(model_id=self.model_id)
        llm = ChatGoogleGenerativeAI(
            model= self.model_id,
            google_api_key= os.getenv("api_key"),
            temperature= os.getenv("temperature", 0.0),
            # top_k= os.getenv("top_k", 20),
            # top_p= os.getenv("top_p", 1),
            # callbacks = [self.llm_cb]
            )
        self.llm = llm
        
    