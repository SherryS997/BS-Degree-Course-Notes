import os
import json
import requests
from mistralai import Mistral

def generate_notes(ocr_text, api_key):
    """Generate notes for the entire OCR text using the Mistral API."""
    model = "mistral-large-latest"
    
    # Initialize the Mistral client with the API key
    client = Mistral(api_key=api_key)

    # Prepare the user message for the Mistral API
    user_message = f"""
```
{ocr_text}
```
Based on the OCR output above, do the following:

You are an exceptional educator. Your task is to create comprehensive, high-quality notes based on the OCR output of scientific texts or slides. Pay special attention to accurately capturing all mathematical formulas and expressions as presented in the OCR text. Follow these guidelines:

1. **Overview**: Start with a brief summary of what will be covered in the notes.

2. **Organizational Structure**: Use clear, hierarchical Markdown formatting:
   - Use `#` for main topics
   - Use `##` for subtopics
   - Use `###` for specific points or examples

3. **Key Areas to Cover**:
   - **Definition and Scope**: Explain the main topics covered.
   - **Historical Context**: Discuss the evolution of the topics.
   - **Applications**: Highlight key applications relevant to the topics.
   - **Fundamental Concepts**: Define important concepts and terminology.
   - **Challenges**: Identify any challenges associated with the topics.
   - **Techniques or Algorithms**: Introduce basic techniques or algorithms discussed.

4. **Content for Each Main Topic**:
   - **Explanation**: Provide clear and detailed explanations.
   - **Examples/Use Cases**: Include relevant examples to illustrate points.
   - **Definitions**: Define any new or important terms introduced.
   - **Tools and Libraries**: Mention any Python libraries or tools that are commonly used in the context.

5. **Mathematical Content**: Ensure that all mathematical formulas, equations, and expressions are captured accurately as they appear in the OCR output. Clearly format these elements for readability and place them in $${{math}}$$.

6. **Visual Elements**: If there are diagrams or visual elements in the OCR text, describe them in detail.

7. **Interpreted Text**: If you encounter unclear or potentially erroneous text in the OCR output, make a reasonable interpretation and note it as [Interpreted from OCR: your interpretation].

8. **Concept Connections**: Throughout the notes, make connections between different concepts to show how they relate to each other.

9. **Summary and Key Takeaways**: After covering all content, include a concise review of the most important points.

10. **Review Questions**: End with a section listing 5-7 thought-provoking questions that encourage deeper understanding of the material.

These notes should serve as a comprehensive resource for someone learning about the topics covered. Make them clear, informative, and engaging, ensuring that all mathematical content is represented accurately and effectively.
    """

    # Create the chat completion request to Mistral API
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ]
    )

    # Handle the response from Mistral API
    if chat_response.choices:
        return chat_response.choices[0].message.content
    else:
        print("Error: No choices returned from the Mistral API response.")
        return None

def read_ocr_output(file_path):
    """Read the OCR output from the given file."""
    with open(file_path, 'r') as file:
        return file.read()

def main():
    # Get API key from environment variable
    api_key = "Hzh0Y7DHnIKijOBKj0gvqHBvgG8xLoTw"

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Input and output file paths
    input_file = "Week 2 LLM.txt"
    output_file = "detailed_llm_notes.md"

    # Read the OCR output
    ocr_text = read_ocr_output(input_file)

    print("Generating notes...")
    # Generate notes for the entire OCR text
    notes = generate_notes(ocr_text, api_key)

    if notes:
        # Save the notes to a markdown file
        with open(output_file, 'w') as file:
            file.write(notes)
        print(f"Detailed notes have been saved to {output_file}")
    else:
        print("Failed to generate notes.")

if __name__ == "__main__":
    main()
