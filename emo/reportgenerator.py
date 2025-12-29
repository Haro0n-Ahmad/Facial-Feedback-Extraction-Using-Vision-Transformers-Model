import requests


def reportgeneratorGPT(tags, emotions_and_timestamp):
    """
    Generates an analytical report using a local LLaMA model via Ollama
    based on intended emotions and detected emotions from a video.
    """

    prompt = (
        f"I have the following intended emotions for a video: {tags}. "
        f"The detected emotions over time are: {emotions_and_timestamp}. "
        f"Analyze whether the detected emotions align with the intended emotions. "
        f"Generate a clear report and suggest improvements if needed."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
    except requests.exceptions.RequestException as e:
        return {
            "error": "Failed to connect to Ollama server",
            "details": str(e)
        }

    if response.status_code == 200:
        try:
            data = response.json()
            return {
                "completion_text": data.get("response", "")
            }
        except ValueError:
            return {
                "error": "Invalid JSON response from Ollama",
                "details": response.text
            }
    else:
        return {
            "error": f"Ollama request failed with status code {response.status_code}",
            "details": response.text
        }
