# auto_programmer.py

import os
from dotenv import load_dotenv
import re
import subprocess
import tempfile
from interface.chatgpt_interface import ChatGPTInterface

MAX_RETRIES = 5


def extract_code(text):
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def run_code(file_path):
    result = subprocess.run(
        ["python3", file_path], capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout, result.stderr


def write_temp_file(code, name_hint="generated_module"):
    temp_dir = os.path.join("/tmp", "wopper_autogen")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{name_hint}.py")
    with open(file_path, "w") as f:
        f.write(code)
    return file_path


def auto_generate_program(requirement):
    chat = ChatGPTInterface()
    prompt = (
        "Write a Python module that satisfies this requirement."
        " Include a test main to verify correctness."
        f"\n\nRequirement: {requirement}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\nAttempt {attempt}...")
        reply = chat.ask(prompt)
        code = extract_code(reply)

        if not code:
            print("No code block found in reply:", reply)
            break

        file_path = write_temp_file(code)
        success, out, err = run_code(file_path)

        if success:
            print("✅ Program passed test:")
            print(out)
            return file_path
        else:
            print("❌ Program failed. Stderr:")
            print(err)
            prompt = (
                "The following program failed to run correctly. Please fix it."
                f"\n\nOriginal Requirement: {requirement}"
                f"\n\nPrevious Code:\n```python\n{code}\n```"
                f"\n\nError Output:\n{err}"
            )

    print("🚫 Failed after multiple attempts.")
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 auto_programmer.py \"<requirement string>\"")
    else:
        auto_generate_program(sys.argv[1])

