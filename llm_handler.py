import os
import json # Import json module
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_VERSION = os.getenv("MODEL_VERSION", "claude-3-haiku-20240307") # Default model

client = OpenAI(api_key=OPENAI_API_KEY)
# Add more client initializations for other LLMs as needed

def summarize_text_with_llm(item_to_summarise: dict, model_provider: str = "openai", model_version: str = "gpt-3.5-turbo-1106") -> dict:
    """
    Summarizes the given feedback item (dictionary) using the specified LLM provider and model.
    The LLM is expected to return a JSON string with sentiment, summary, and constructiveCriticism.

    Raises:
        ValueError: If the specified model_provider is not supported or its API key is missing.
        NotImplementedError: If the summarization logic for the provider is not implemented.
    """

    llm_output_dict = {
        "response": "Could not process feedback."
    }

    system_prompt = os.getenv("SYSTEM_PROMPT")

    # Construct the user message from the input dictionary
    anon_identifier = item_to_summarise.get('anon_identifier', 'Tidak disebutkan')
    context_text = item_to_summarise.get('context_text', '')
    feedback_text = item_to_summarise.get('feedback_text', '')

    # For Stefani, treat feedback_text as the user's question or code, and context as conversation context if any
    user_message_content = feedback_text


    if model_provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        selected_model = model_version
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_content}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            response_text = response.choices[0].message.content
            llm_output_dict["response"] = response_text.strip()
        except Exception as e:
            print(f"Error during OpenAI completion: {e}")
            llm_output_dict["response"] = f"Error: {e}"
    elif model_provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("Google Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.")
        raise NotImplementedError("Google Gemini summarization logic is not yet implemented for dictionary I/O.")
    else:
        raise ValueError(f"Unsupported LLM provider: {model_provider}. Supported providers are 'openai', 'gemini'.")

    return llm_output_dict

if __name__ == "__main__":
    # Example usage for Stefani as a C programming tutor
    sample_input = {
        "anon_identifier": "user-1234",
        "context_text": "",  # conversation context if any
        "feedback_text": "Gimana cara bikin loop di C?"
    }

    print("Testing Stefani LLM handler...")
    try:
        result = summarize_text_with_llm(sample_input, model_provider="openai")
        print("\nLLM Response:")
        print(result["response"])
    except Exception as e:
        print(f"Error: {e}")

