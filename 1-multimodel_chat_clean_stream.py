import json
import boto3
import streamlit as st
from botocore.config import Config
import re

# Model configuration
MODEL_OPTIONS = {
    "Nova Micro": "us.amazon.nova-micro-v1:0",
    "Nova Lite": "us.amazon.nova-lite-v1:0",
    "Nova Pro": "us.amazon.nova-pro-v1:0",
    "Nova Lite Reasoning": "arn:aws:bedrock:us-east-1:963067361214:provisioned-model/dx9g06qbz7f6",
}

# App configuration
st.set_page_config(page_title="Amazon Bedrock Reasoning Capability Test Platform", layout="wide")
st.title("💬 Amazon Bedrock Reasoning Capability Test Platform")

def check_missing_solution_end(text):
    """
    Check specifically for missing end_of_solution tag when begin_of_solution is present
    """
    begin_solution = "<|begin_of_solution|>" in text
    end_solution = "<|end_of_solution|>" in text
    
    return begin_solution and not end_solution

def format_streaming_response(text, is_reasoning_prompt=True):
    """
    Formats the response for streaming, handling partial tags and incomplete content
    """
    if not is_reasoning_prompt:
        return text + "▌"
        
    # Initialize formatted text
    formatted_text = ""
    current_text = text
    
    # Check if thought section has started
    if "<|begin_of_thought|>" in current_text:
        formatted_text += "### Note Lite Reasoning for you - I am not perfect, but will try my best\n\n"
        # Split at the beginning of thought
        parts = current_text.split("<|begin_of_thought|>", 1)
        if len(parts) > 1:
            thought_content = parts[1]
            # Check if thought section has ended
            if "<|end_of_thought|>" in thought_content:
                # Get content before end tag
                thought_content = thought_content.split("<|end_of_thought|>")[0]
                current_text = current_text.split("<|end_of_thought|>", 1)[-1]
            formatted_text += thought_content + "\n\n"
    
    # Check if solution section has started
    if "<|begin_of_solution|>" in current_text:
        formatted_text += "### Solution\n\n"
        # Split at the beginning of solution
        parts = current_text.split("<|begin_of_solution|>", 1)
        if len(parts) > 1:
            solution_content = parts[1]
            # Check if solution section has ended
            if "<|end_of_solution|>" in solution_content:
                # Get content before end tag
                solution_content = solution_content.split("<|end_of_solution|>")[0]
            formatted_text += solution_content
    
    # If no sections have started yet, just return the text
    if formatted_text == "":
        return current_text + "▌"
    
    # Check for missing solution end tag and add warning if needed
    if is_reasoning_prompt and check_missing_solution_end(text):
        formatted_text += "\n\n⚠️ *Note: Response was cut off due to token limit. Some sections may be incomplete.*"
        
    return formatted_text.strip() + "▌"

def format_final_response(text, is_reasoning_prompt=True):
    """
    Formats the final complete response, removing all tags
    """
    if not is_reasoning_prompt:
        return text
        
    # Extract thought content
    thought_match = re.search(r'<\|begin_of_thought\|>(.*?)(?:<\|end_of_thought\|>|$)', text, re.DOTALL)
    thought_content = thought_match.group(1).strip() if thought_match else ""

    # Extract solution content
    solution_match = re.search(r'<\|begin_of_solution\|>(.*?)(?:<\|end_of_solution\|>|$)', text, re.DOTALL)
    solution_content = solution_match.group(1).strip() if solution_match else ""

    # Format with headers
    formatted_text = ""
    if thought_content:
        formatted_text += "### Nova Lite Reasoning for you - I still learning, but will try my best\n\n" + thought_content + "\n\n"
    if solution_content:
        formatted_text += "### Solution\n\n" + solution_content

    # Check for missing solution end tag and add warning if needed
    if is_reasoning_prompt and check_missing_solution_end(text):
        formatted_text += "\n\n⚠️ *Note: Response was cut off due to token limit. Some sections may be incomplete.*"

    return formatted_text.strip() if formatted_text else text

[rest of the file remains exactly the same from the previous artifact]
