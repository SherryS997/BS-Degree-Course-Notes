import os
import json
import asyncio
import base64
from pdf2image import convert_from_path
import logging
from mistralai import Mistral
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
from difflib import SequenceMatcher  # Use for comparing similarity between slides

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
Please perform OCR on the provided scientific text or slides and convert the extracted content into a detailed markdown format, ensuring accuracy, formatting, formulas, etc.
                    """},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ]
            }
        ]

        response = client.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during OCR from Mistral: {str(e)}")
        return None

# Function to compare slides and check if the current one has new content
def is_new_content(current_slide, reference_slide, threshold=0.9):
    """Returns True if current_slide has content not in reference_slide."""
    matcher = SequenceMatcher(None, current_slide, reference_slide)
    similarity_ratio = matcher.ratio()
    return similarity_ratio < threshold

# Function to process a single page
async def process_page(pdf_file, pdf_path, page_num, api_key):
    output_image_path = f"{pdf_path.replace('.pdf', '')}_{page_num}.jpg"
    
    # Convert PDF page to Image
    image_path = await convert_pdf_page_to_image(pdf_path, page_num, output_image_path)
    if not image_path:
        return page_num, None

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        os.remove(image_path)
        return page_num, None

    # Get OCR from Mistral API
    ocr_response = await get_ocr_from_mistral(image_base64, api_key)
    
    os.remove(image_path)  # Clean up temporary image
    if not ocr_response:
        return page_num, None

    return page_num, ocr_response

# Function to process a single PDF, starting from the last slide
async def process_pdf(pdf_file, pdf_folder, api_key):
    pdf_path = os.path.join(pdf_folder, pdf_file)
    logging.info(f"Processing {pdf_file}")

    # Get number of pages
    num_pages = len(convert_from_path(pdf_path, dpi=72))

    results = []
    reference_content = None  # This will hold the content of the reference slide

    # Process slides from last to first
    for page_num in range(num_pages, 0, -1):
        page_num, current_content = await process_page(pdf_file, pdf_path, page_num, api_key)
        
        if not current_content:
            continue

        # Check if the current slide has new content compared to the reference
        if reference_content is None or is_new_content(current_content, reference_content):
            # New content, save it and make it the reference for earlier slides
            reference_content = current_content
            results.append(f"# {pdf_file} - Page {page_num}\n\n{current_content}\n\n")
        else:
            logging.info(f"Skipping page {page_num} as it is similar to a later slide.")

    return results[::-1]  # Reverse the results back to original order

# Function to process multiple PDFs and save output
async def process_pdfs(pdf_folder, api_key):
    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.endswith('.pdf')])

    for pdf_file in pdf_files:
        output_txt_file = f"NLP/{pdf_file.replace('.pdf', '.txt')}"
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
    MISTRAL_API_KEY = "Hzh0Y7DHnIKijOBKj0gvqHBvgG8xLoTw"
    
    pdf_folder = "pdfs/NLP"
    await process_pdfs(pdf_folder, MISTRAL_API_KEY)
    print(f"Markdown outputs saved for each PDF file.")

if __name__ == "__main__":
    asyncio.run(main())
