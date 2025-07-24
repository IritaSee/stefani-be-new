import os
import json # Import json module
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_VERSION = os.getenv("MODEL_VERSION", "claude-3-haiku-20240307") # Default model


# Initialize OpenAI client if API key is available
openai_client = None
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Placeholder for specific LLM client initializations
# For example, if using OpenAI:
# from openai import OpenAI
# if OPENAI_API_KEY:
#     openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Add more client initializations for other LLMs as needed

def summarize_text_with_llm(item_to_summarise: dict, model_provider: str = "openai", model_version: str = "gpt-3.5-turbo-1106") -> dict:
    """
    Summarizes the given feedback item (dictionary) using the specified LLM provider and model.
    The LLM is expected to return a JSON string with sentiment, summary, and constructiveCriticism.

    Args:
        item_to_summarise: A dictionary containing the feedback details.
                           Expected keys: 'anon_identifier', 'context_text', 'feedback_text'.
        model_provider: The LLM provider to use (e.g., "anthropic", "openai", "gemini").
                        Defaults to "anthropic".
        model_version: The specific model version to use.
                       Defaults to the MODEL_VERSION environment variable or "claude-3-haiku-20240307".

    Returns:
        A dictionary parsed from the LLM's JSON response, typically containing:
        {'sentiment': str, 'summary': str, 'constructiveCriticism': str}.
        Returns a dictionary with an 'error' key in case of issues.

    Raises:
        ValueError: If the specified model_provider is not supported or its API key is missing.
        NotImplementedError: If the summarization logic for the provider is not implemented.
    """

    llm_output_dict = {
        "sentiment": "Error",
        "summary": "Could not process feedback.",
        "constructiveCriticism": "No specific details available."
    }

    system_prompt = """
    You are an HR manager, reading candid feedback to an employee. This feedback was gathered anonymously, though some may have included names.
    The feedback consists of three parts:
    1. How the feedback giver knows the person or context of their interaction.
    2. Candid feedback about the person (this can be very direct, emotional, or use informal language).
    3. Additional context for the feedback provided in the second part.

    Your task is to summarize this feedback. Ensure the summary is constructive, maintains factual accuracy (including mistakes mentioned), but is delivered in a way that does not cause undue distress. Your response should be in a relaxed, colloquial Bahasa Indonesia, as if speaking to a colleague.

    Your output MUST be a valid JSON object with the following three keys:
    - "sentiment": (string) Analyze the sentiment of the feedback (e.g., "Positif Banget", "Agak Negatif", "Netral Aja").
    - "summary": (string) A summary of the feedback, maintaining facts and mistakes, in a supportive tone and colloquial Bahasa Indonesia.
    - "constructiveCriticism": (string) Constructive advice based on the feedback, also in colloquial Bahasa Indonesia.

    Example of the input you will receive (the actual content will vary):
    Pengenal Anonim: Teman satu tim proyek X
    Konteks Feedback: Saat presentasi mingguan
    Isi Feedback: Presentasinya bagus banget, tapi slide-nya kebanyakan tulisan, bikin ngantuk.

    Example of your desired JSON output:
    {
      "sentiment": "Netral Aja",
      "summary": "Feedbacknya bilang presentasi kamu udah oke, tapi slide-nya terlalu banyak teks jadi bikin audience agak bosen.",
      "constructiveCriticism": "Coba deh slide presentasinya dibikin lebih visual, mungkin bisa pake gambar atau poin-poin aja biar lebih engaging."
    }
    """

    # Construct the user message from the input dictionary
    anon_identifier = item_to_summarise.get('anon_identifier', 'Tidak disebutkan')
    context_text = item_to_summarise.get('context_text', 'Tidak disebutkan')
    feedback_text = item_to_summarise.get('feedback_text', 'Input feedback kosong.')
    
    user_message_content = f"Pengenal Anonim: {anon_identifier}\nKonteks Feedback: {context_text}\nIsi Feedback: {feedback_text}"


    if model_provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        selected_model = model_version
        try:
            response = openai.ChatCompletion.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_content}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            response_text = response["choices"][0]["message"]["content"]
            # Attempt to parse the LLM response as JSON
            try:
                # Clean the response text if it's wrapped in markdown code blocks
                if response_text.strip().startswith("```json"):
                    response_text = response_text.strip()[7:-3].strip()
                elif response_text.strip().startswith("```"):
                    response_text = response_text.strip()[3:-3].strip()
                parsed_llm_response = json.loads(response_text)
                if all(key in parsed_llm_response for key in ["sentiment", "summary", "constructiveCriticism"]):
                    llm_output_dict = parsed_llm_response
                else:
                    llm_output_dict["summary"] = "LLM response did not contain all expected JSON keys."
                    llm_output_dict["constructiveCriticism"] = f"Raw LLM response: {response_text}"
            except json.JSONDecodeError as je:
                print(f"Error decoding JSON from LLM response: {je}")
                llm_output_dict["summary"] = "Failed to parse LLM response as JSON."
                llm_output_dict["constructiveCriticism"] = f"Raw LLM response: {response_text}. Error: {je}"
        except Exception as e:
            print(f"Error during OpenAI summarization: {e}")
            llm_output_dict["summary"] = "Could not summarize text with OpenAI due to an API error."
            llm_output_dict["constructiveCriticism"] = str(e)
    elif model_provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("Google Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.")
        raise NotImplementedError("Google Gemini summarization logic is not yet implemented for dictionary I/O.")
    else:
        raise ValueError(f"Unsupported LLM provider: {model_provider}. Supported providers are 'openai', 'gemini'.")

    return llm_output_dict

if __name__ == "__main__":
    # Example of how to use the summarizer with a dictionary input
    sample_feedback_item = {
        "anon_identifier": "Rekan kerja di tim Alpha",
        "context_text": "Waktu meeting mingguan kemarin lusa",
        "feedback_text": "Menurutku sih si X ini idenya oke-oke, tapi cara nyampeinnya kadang muter-muter jadi orang bingung. Terus suka telat juga kalo janjian."
    }

    print("Attempting to summarize feedback item...")

    # Ensure OPENAI_API_KEY is set in your .env file and openai library is installed.
    try:
        summary_result = summarize_text_with_llm(sample_feedback_item, model_provider="openai")
        print("\nLLM Processing Result:")
        print(json.dumps(summary_result, indent=2, ensure_ascii=False))
    except (ValueError, NotImplementedError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\nTo test thoroughly, ensure API keys are in .env, necessary libraries are installed,")
    print("and that the LLM provider ('openai') is correctly implemented.")

