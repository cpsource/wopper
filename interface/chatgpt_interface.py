# Thin wrapper around the OpenAI ChatGPT API.
import os
from dotenv import load_dotenv
from openai import OpenAI
from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting chatgpt_interface.py")

class ChatGPTInterface:
    def __init__(self, model="gpt-4o"):
        # Load .env and read API key
        load_dotenv(os.path.expanduser("~/.env"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in ~/.env")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def ask(self, prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error querying ChatGPT API: {e}]"

# ---------------------------------------
# 🧪 Local test
# ---------------------------------------
if __name__ == "__main__":
    bot = ChatGPTInterface()
    question = "What are some common types of grocery stores?"
    reply = bot.ask(question)
    log.info("Response:\n%s", reply)

