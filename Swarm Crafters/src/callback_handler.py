"""Callback Handler that prints to std out."""
import threading
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

GEMINI_COST_MAP : dict = { "gemini-2.0-flash" : [0.10,0.40],
                           "gemini-2.0-flash-lite" : [0.075,0.30]}

class GoogleCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks gemini info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(self,model_id) -> None:
        super().__init__()
        self.model_id=model_id

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        print(token)


    @staticmethod
    def calculate_cost(no_of_input_tokens, no_of_output_tokens,model_id):
        input_token_price, output_token_price = GEMINI_COST_MAP.get(model_id)

        return ((no_of_input_tokens * input_token_price) + (no_of_output_tokens * output_token_price)) / 1000000 #calculates cost per million tokens

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # compute tokens and cost for this request
        if response.generations and response.generations[0] and response.generations[0][0] and response.generations[0][0].message:
            message= response.generations[0][0].message
            if message.usage_metadata :
                usage = message.usage_metadata
                no_of_input_tokens = usage.get("input_tokens", None)
                no_of_output_tokens = usage.get("output_tokens", None)
                total_tokens = usage.get("total_tokens", None)
                total_cost = self.calculate_cost(no_of_input_tokens, no_of_output_tokens,self.model_id)
                prompt_tokens = no_of_input_tokens
                completion_tokens = no_of_output_tokens
                # update shared state behind lock
                self.total_cost += total_cost
                self.total_tokens += total_tokens
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.successful_requests += 1
            
    