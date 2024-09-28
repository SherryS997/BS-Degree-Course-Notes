import os
import json
import asyncio
import base64
from pdf2image import convert_from_path
import logging
from mistralai import Mistral
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm  # Use tqdm's asyncio version for progress bars

# Set up logging
logging.basicConfig(filename='pdf_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to convert a single PDF page to image
async def convert_pdf_page_to_image(pdf_path, page_num, output_image_path, dpi=300):
    try:
        images = await asyncio.to_thread(convert_from_path, pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
        if images:
            await asyncio.to_thread(images[0].save, output_image_path)
            return output_image_path
        else:
            logging.warning(f"No image generated for page {page_num} of {pdf_path}")
            return None
    except Exception as e:
        logging.error(f"Error converting page {page_num} of {pdf_path}: {str(e)}")
        return None

# Function to encode image to base64
def encode_image_to_base64(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {str(e)}")
        return None

# Function to get OCR from Mistral API
async def get_ocr_from_mistral(image_base64, api_key):
    try:
        model = "pixtral-12b-2409"
        client = Mistral(api_key=api_key)

        # Define the message structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
Please perform OCR on the provided scientific text or slides and convert the extracted content into a detailed markdown format. Ensure the following:

1. **Accuracy**: Pay extra attention to scientific terms, symbols, and formulas, ensuring they are correctly captured.
2. **Formatting**: 
   - Encode section titles, headings, and subheadings using proper markdown syntax (`#`, `##`, etc.).
   - Maintain paragraph structure, and use bullet points (`-`) or numbered lists where applicable.
   - Ensure all **bold** and *italicized* text is correctly formatted.
3. **Formulas and Equations**: Use inline code or block code (using ```math```) to format scientific formulas and equations.
4. **Diagrams and Images**: For images, graphs, or diagrams, use markdown image syntax `![]()` with placeholders if OCR can't capture them directly.
5. **Tables**: Properly format tables using markdown table syntax (`|` and `---`).
6. **Code Blocks**: Wrap any code snippets or special notation using appropriate markdown code blocks ``` ``` with the correct language identifier.
7. **Symbols**: Ensure that special characters or symbols (e.g., Greek letters, mathematical operators) are represented accurately.
8. **Multilingual Content**: If there are sections with different languages, identify them appropriately in the output.

Output all the extracted text in markdown format and ensure the scientific integrity of the content.                    
                    """},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ]
            }
        ]

        # Send the request to Mistral API
        response = client.chat.complete(model=model, messages=messages)

        # Extract and return the content
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during OCR from Mistral: {str(e)}")
        return None

# Function to process a single page
async def process_page(pdf_file, pdf_path, page_num, api_key):
    output_image_path = f"{pdf_path.replace('.pdf', '')}_{page_num}.jpg"
    
    # Step 1: Convert PDF page to Image
    image_path = await convert_pdf_page_to_image(pdf_path, page_num, output_image_path)
    if not image_path:
        return page_num, None

    # Step 2: Encode image to base64
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        os.remove(image_path)
        return page_num, None

    # Step 3: Get OCR from Mistral API
    ocr_response = await get_ocr_from_mistral(image_base64, api_key)
    
    # Clean up the temporary image file
    os.remove(image_path)
    
    if not ocr_response:
        return page_num, None

    # Step 4: Format the markdown output
    return page_num, f"# {pdf_file} - Page {page_num}\n\n{ocr_response}\n\n"

# Function to process a single PDF with tqdm progress bar
async def process_pdf(pdf_file, pdf_folder, api_key):
    pdf_path = os.path.join(pdf_folder, pdf_file)
    logging.info(f"Processing {pdf_file}")
    
    # Get the number of pages in the PDF
    num_pages = len(convert_from_path(pdf_path, dpi=72))  # Use low DPI just for counting pages

    # Use tqdm to display progress for each page
    tasks = [process_page(pdf_file, pdf_path, page_num, api_key) for page_num in range(1, num_pages + 1)]
    results = []
    
    # Progress bar will track page processing
    async for result in tqdm(asyncio.as_completed(tasks), total=num_pages, desc=f"Processing {pdf_file}"):
        results.append(await result)
    
    return [result[1] for result in sorted(results, key=lambda x: x[0])]

# Function to process multiple PDFs with individual output files
async def process_pdfs(pdf_folder, api_key):
    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.endswith('.pdf')])

    # Process each PDF sequentially
    for pdf_file in pdf_files:
        # Save each PDF result in its own markdown file
        output_txt_file = f"DLCV/{pdf_file.replace('.pdf', '.txt')}"
        
        if os.path.exists(output_txt_file):
            continue
        
        pdf_results = await process_pdf(pdf_file, pdf_folder, api_key)
        
        with open(output_txt_file, 'w') as out_file:
            for result in pdf_results:
                if result:
                    out_file.write(result)
        logging.info(f"Markdown output saved to {output_txt_file}")

# Main function
async def main():
    # Get API key from environment variable
    MISTRAL_API_KEY = "Hzh0Y7DHnIKijOBKj0gvqHBvgG8xLoTw"
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    # Specify the folder containing PDF files
    pdf_folder = "pdfs/DLCV"
    
    await process_pdfs(pdf_folder, MISTRAL_API_KEY)
    print(f"Markdown outputs saved for each PDF file.")

# Example usage
if __name__ == "__main__":
    asyncio.run(main())
