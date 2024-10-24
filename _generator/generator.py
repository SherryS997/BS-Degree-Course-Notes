import os
import json
from mistralai import Mistral
from tqdm import tqdm
import requests
import time
import google.generativeai as genai

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

def check_relevance(ocr_text, previous_notes, api_key):
    """Strictly check if the OCR text contains new and relevant information using a detailed template."""
    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)

    user_message = f"""
Previous notes:
```
{previous_notes}
```

New OCR text:
```
{ocr_text}
```

Analyze the OCR text above in relation to the previous notes. Determine if this page contains new, relevant, and non-redundant information. Be extremely strict in your evaluation.

Provide your response in the following strict template format:

NEW_CONCEPTS: [0-5]
DEPTH_ADDED: [0-5]
RELEVANCE: [0-5]
CLARITY: [0-5]
ORIGINALITY: [0-5]

Guidelines for scoring:
- NEW_CONCEPTS: Number of entirely new concepts introduced (0 = none, 5 = many new concepts)
- DEPTH_ADDED: How much depth is added to existing ideas (0 = no new depth, 5 = significant new insights)
- RELEVANCE: How directly related the content is to the primary subject (0 = unrelated, 5 = highly relevant)
- CLARITY: The clarity and coherence of the text (0 = incomprehensible, 5 = perfectly clear and well-structured)
- ORIGINALITY: The degree of novelty and non-redundancy (0 = highly redundant, 5 = no repetition or overlap with previous content)

Your response should contain only these five lines with numerical scores. Do not include any additional explanation or text.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in content analysis. Your task is to critically evaluate new content for relevance and novelty, providing strict numerical scores in a template format."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }

    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    while True:
        try:
            # response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=data)
            response = model.generate_content(user_message)
            
            # if response.status_code == 200:
            if True:
                # result = response.json()['choices'][0]['message']['content'].strip()
                result = response.text.strip()
                
                # Parse the result
                lines = result.split('\n')
                if len(lines) != 5:
                    print(lines)
                    print("Error: Unexpected response format")
                    return False

                scores = {}
                for line in lines:
                    key, value = line.split(': ')
                    scores[key] = int(value)

                # Calculate the overall relevance score
                overall_score = (
                    scores['NEW_CONCEPTS'] * 1.5 +  # Weight new concepts more heavily
                    scores['DEPTH_ADDED'] +  # Also prioritize depth
                    scores['RELEVANCE'] * 1.5 +  # Ensure strong subject relevance
                    scores['CLARITY'] +
                    scores['ORIGINALITY'] * 2
                ) / 7  # Normalize to a 0-5 scale

                # Set a very high threshold for relevance
                is_relevant = overall_score > 3 and min(scores.values()) >= 1

                print(f"Relevance check result:\n{result}")
                print(f"Overall score: {overall_score:.2f}")
                print(f"Final decision: {'Relevant' if is_relevant else 'Not relevant'}")

                return is_relevant
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"Request failed: {e}")
        
            # Wait for 5 seconds before trying again
            time.sleep(5)


def generate_notes(ocr_text, previous_notes, api_key, domain="LLM", detail_level="very detailed"):
    """
    Generate notes for a single page of OCR text using the Mistral API, with options for domain and customization.
    
    Parameters:
    - ocr_text: The OCR text extracted from the document.
    - previous_notes: Existing notes to avoid redundancy.
    - api_key: The API key for Mistral or another language model.
    - domain: Optional. Specify the domain or subject area (e.g., "Mathematics", "History"). Defaults to None for general use.
    - detail_level: Customize the level of detail. Options are "detailed" or "summary". Defaults to "detailed".
    """
    
    # Initialize Mistral or other model client
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)

    # Create the user message with flexibility for different domains
    user_message = f"""
Previous notes:
```
{previous_notes}
```

New OCR text:
```
{ocr_text}
```
Based on the OCR output above and the previous notes, creating comprehensive, high-quality notes. Follow these guidelines:

1. Content Focus:
   - Focus solely on the new, substantive content presented in the OCR text.
   - Do not include introductions, conclusions, metadata, or contact information.
   - Ignore formatting instructions, placeholder content, or generic structures.
   - If the text relates to specific domain (like {domain}), ensure that the content is aligned with the topic.

2. Structure:
   - Use Markdown headings: # for main topics, ## for subtopics, ### for specific points.
   - Only create new headings if they introduce genuinely new topics not covered in previous notes.

3. Mathematical Content (if applicable):
   - For technical subjects such as Mathematics or Engineering, format inline mathematical expressions with single dollar signs: $expression$.
   - For block mathematical expressions, use double dollar signs:
     $$
     expression
     $$
   - Skip this instruction if the domain is non-technical (e.g., History or Literature).

4. Visuals and Diagrams:
   - If a crucial visual or diagram is mentioned, briefly describe its content in text form. 
   - Include placeholders for visuals when critical to understanding.

5. Avoiding Repetition:
   - Do not repeat any information already covered in the previous notes.
   - If the entire page doesn't add any new information, respond with only the word 'None'.
   - If minor additional info exists, flag it as "[Partially Redundant: minor additions]".

6. Customization ({detail_level}):
   - For "detailed" notes, provide clear explanations for new terms and concepts, and include examples to illustrate new points.
   - For "summary" notes, keep content brief, only listing key points without detailed explanations or examples.

7. Non-Standard OCR Elements:
   - If the OCR text contains tables, lists, or non-standard structures, interpret and reformat them as appropriate.
   - For unclear content, make a reasonable interpretation and note it as [Interpreted: your interpretation].

8. Clarity and Completeness:
   - Ensure all new content is self-contained and easy to understand without external references.
   - Provide detailed and complete notes, summarizing only relevant new information. 
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert note-taker across multiple domains, capable of creating clear, high-quality notes."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }

    model = genai.GenerativeModel("gemini-1.5-flash-002")  # Use the relevant model for your needs

    while True:
        try:
            # response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=data)
            response = model.generate_content(user_message)  # Generating content from the model
            
            # if response.status_code == 200:
            if True:  # For simplicity
                return response.text  # Return the generated notes
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"Request failed: {e}")
            # Wait for 5 seconds before trying again
            time.sleep(5)

def read_ocr_output(file_path):
    """Read the OCR output from the given file."""
    with open(file_path, 'r') as file:
        return file.read()

def main():
    api_key = "Hzh0Y7DHnIKijOBKj0gvqHBvgG8xLoTw"  # Replace with your actual API key
    if not api_key:
        raise ValueError("API key not found")

    input_files = [
        "LLM/Week 2 LLM.txt",
        # "LLM/W2Lec6-Stemming-Lemmatization.txt",
        # "LLM/W2Lec7-MorphologicalAnalysis.txt",
    ]
    output_file = "detailed_llm_week2_notes.md"

    all_notes = ""
    total_pages = sum(len(read_ocr_output(file).split("- Page ")) - 1 for file in input_files)

    with tqdm(total=total_pages, desc="Generating Notes") as pbar:
        for input_file in input_files:
            print(f"\nProcessing {input_file}...")
            ocr_text = read_ocr_output(input_file)
            
            # Split the OCR text into pages
            pages = ocr_text.split("- Page ")
            
            for i, page in enumerate(pages[1:], 1):  # Skip the first split as it's empty
                page_text = f"- Page {page}"  # Reconstruct the page marker
                
                # Step 1: Check relevance
                if check_relevance(page_text, all_notes, api_key):
                    # Step 2: Generate notes
                    notes = generate_notes(page_text, all_notes, api_key)
                    
                    if notes and 'none' not in notes.lower():
                        all_notes += notes + "\n\n"
                        # Save the notes after each page is processed
                        with open(output_file, 'w') as file:
                            file.write(all_notes)
                    else:
                        print(f"  No new information on page {i} of {input_file}.")
                else:
                    print(f"  Skipping page {i} of {input_file} - content not relevant or new.")
                
                pbar.update(1)

        print(f"\nFinished processing all files. Notes saved to {output_file}")

if __name__ == "__main__":
    main()

