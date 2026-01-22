import subprocess


def generate_answer(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.stdout.decode("utf-8", errors="ignore").strip()
