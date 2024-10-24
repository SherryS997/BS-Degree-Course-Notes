import os
import requests, time
from tqdm import tqdm
import google.generativeai as genai

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

def clean_notes(raw_notes, api_key):
    """Clean the entire notes using the LLM API, ensuring no rewriting or summarization occurs."""

    # Define explicit instructions to avoid rewriting
    instructions = """
Clean and format the above notes. The goal is to only remove useless information, redundant sections, and clean up the formatting.

**Important Rules**:
1. **Do NOT rewrite, paraphrase, or change the meaning** of any text.
2. Only focus on **removing repeated or redundant or irrelevant content**, or **rearranging** content when necessary for clarity.
3. Do not add new content, summaries, or conclusions. Keep all original phrasing.
4. **Formatting**:
   - Headings: Use `#` for main topics, `##` for subtopics, and avoid `###` unless strictly needed.
   - Lists should use `*` for bullet points or numbers for ordered lists. Avoid unnecessary `####` headers for points.
   - Code blocks (` ``` `), math expressions (`$`, `$$`), and lists must have an empty line before them.
   - Remove any references or links to images or external sources.
5. **No Summaries**:
   - The output should **only** contain the cleaned version of the text. Do not add summaries, conclusions, or interpretations.
"""

    # Create the prompt with the text first and instructions after
    prompt = f"""
Text to Clean:
{raw_notes}

Instructions:
{instructions}
"""

    # Send the request to the LLM API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # while True:
    #     try:
    #         # Post the request to the LLM API and handle the response
    #         response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=data)
            
    #         if response.status_code == 200:
    #             return response.json()['choices'][0]['message']['content']
    #         else:
    #             print(f"Error: {response.status_code} - {response.text}")
        
    #     except requests.exceptions.RequestException as e:
    #         print(f"Request failed: {e}")
        
    #     # Wait for 5 seconds before retrying
    #     time.sleep(5)

    model = genai.GenerativeModel("gemini-1.5-flash-002")
    response = model.generate_content(prompt)
    return response.text

def generate_title(cleaned_notes, api_key):
    """Generate a concise title for the notes using the LLM."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are an expert title generator."},
            {"role": "user", "content": f"{cleaned_notes}\n\nGenerate a title with less than 5 words for the above notes:"}
        ]
    }

    # while True:
    #     try:
    #         response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=data)
            
    #         if response.status_code == 200:
    #             return response.json()['choices'][0]['message']['content']
    #         else:
    #             print(f"Error: {response.status_code} - {response.text}")
        
    #     except requests.exceptions.RequestException as e:
    #         print(f"Request failed: {e}")
        
    #     # Wait for 5 seconds before trying again
    #     time.sleep(5)

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(f"{cleaned_notes}\n\nGenerate a title with less than 5 words for the above notes:")
    return response.text

def main():
    api_key = "Hzh0Y7DHnIKijOBKj0gvqHBvgG8xLoTw"  # Replace with your actual API key
    if not api_key:
        raise ValueError("API key not found")
    
    input_file = "detailed_llm_week2_notes.md"
    output_file = "cleaned_notes.md"
    
    # Read the generated notes
    with open(input_file, 'r') as file:
        raw_notes = file.read()
    
    # Clean the entire notes in one go
    cleaned_notes = clean_notes(raw_notes, api_key)
    
    # Generate the title based on the cleaned notes
    title = generate_title(cleaned_notes, api_key)
    
    if title:
        cleaned_notes = f"---\ntitle: {title}\n---\n\n" + cleaned_notes
    
    # Save the cleaned notes to a file
    with open(output_file, 'w') as file:
        file.write(cleaned_notes)
    
    print(f"Finished cleaning notes. Output saved to {output_file}")

if __name__ == "__main__":
    main()
