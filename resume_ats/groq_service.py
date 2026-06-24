from groq import Groq
import os

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
print(os.getenv("GROQ_API_KEY"))
def generate_response(prompt: str) -> str:

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content