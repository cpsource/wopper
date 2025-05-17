# auto_programmer.py
# Iteratively asks ChatGPT to produce a working Python module.

import os
from dotenv import load_dotenv
import sys
import subprocess
import tempfile
import traceback
from interface.chatgpt_interface import ChatGPTInterface
from logger import get_logger

log = get_logger(__name__)
log.debug("Starting auto_programmer.py")

MAX_ATTEMPTS = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "utils")

def generate_code(prompt):
    chatgpt = ChatGPTInterface()
    response = chatgpt.client.chat.completions.create(
        model=chatgpt.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    content = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 'Unknown'
    return content, tokens_used

def extract_code(text):
    # If code is wrapped in triple backticks, extract that block
    if "```" in text:
        lines = text.split("```")
        for block in lines:
            if block.strip().startswith("python"):
                return block.strip().split("\n", 1)[1]  # remove 'python' tag
            elif block.strip().startswith("#") or "def" in block:
                return block.strip()
    return text.strip()

def run_code(path):
    try:
        result = subprocess.run(["python3", path], capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def auto_program(filename, requirement):
    filename = filename.strip().rstrip(",")  # <-- fix trailing comma
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_tokens = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log.info(f"Attempt {attempt}...")
        response, tokens = generate_code(
            f"Write a Python program for the following requirement. Be sure to include a main() that tests the function: {requirement}"
        )
        total_tokens += tokens if isinstance(tokens, int) else 0

        code = extract_code(response)

        # Write to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            tmp.write(code.encode("utf-8"))
            tmp_path = tmp.name

        log.debug("Testing generated code...")
        passed, output = run_code(tmp_path)

        if passed:
            log.info("Test passed. Saving to utils/...\n")
            output_path = os.path.join(OUTPUT_DIR, filename)
            with open(output_path, "w") as f:
                f.write(code)
            os.remove(tmp_path)
            log.info(f"Saved to: {output_path}")
            log.info(f"Total tokens used: {total_tokens}")
            return
        else:
            log.warning("Test failed:")
            log.warning(output)
            new_prompt = f"The following code failed with this error. Fix it and return a working version:\n\n{code}\n\nERROR:\n{output}"
            response, tokens = generate_code(new_prompt)
            total_tokens += tokens if isinstance(tokens, int) else 0

    log.error("❌ All attempts failed. Giving up.")
    log.info(f"Total tokens used: {total_tokens}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        log.error("Usage: python3 auto_programmer.py <filename.py> \"<requirement string>\"")
        sys.exit(1)

    filename = sys.argv[1]
    requirement = sys.argv[2]
    auto_program(filename, requirement)

