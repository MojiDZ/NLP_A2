# chat_assistant.py
from openai import OpenAI


class ChatAssistant:
    def __init__(self, api_key: str):
        self.model = OpenAI(
            api_key=api_key,
            base_url='https://api.llama-api.com'
        )

    def build_api_request(self, prompt: str) -> list:
        return [
            'llama-70b-chat',
            [
                {"role": "system",
                 "content": "Assistant is a large language model that addresses every topic within the "
                            "input text."},
                {"role": "user", "content": prompt},
            ]
        ]

    def generate_answer(self, slice_text: str) -> str:
        request = self.build_api_request(slice_text)

        try:
            response = self.model.chat.completions.create(model=request[0], messages=request[1])
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"
