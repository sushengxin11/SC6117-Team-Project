import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not detected"
    )

client = OpenAI(api_key=api_key)


def call_llm_raw(
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4o-mini",
        max_output_tokens: int = 800,
) -> str:
    """The basic LLM call wrapper, returning plain text output."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_output_tokens,
            temperature=0.7,
        )

        # Get the returned text content
        content = response.choices[0].message.content
        return content

    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""

