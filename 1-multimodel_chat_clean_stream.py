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

def check_missing_tags(text):
    """
    Check for missing end tags when their corresponding begin tags are present
    """
    begin_thought = "<|begin_of_thought|>" in text
    end_thought = "<|end_of_thought|>" in text
    begin_solution = "<|begin_of_solution|>" in text
    end_solution = "<|end_of_solution|>" in text
    
    missing_thought = begin_thought and not end_thought
    missing_solution = begin_solution and not end_solution
    
    return missing_thought or missing_solution

def format_latex(text):
    """
    Format LaTeX expressions for proper rendering in Streamlit with enhanced readability
    """
    def replace_math_symbols(match):
        expr = match.group(1)
        # Common mathematical symbols and their readable forms
        expr = re.sub(r'\\cup', '∪', expr)  # union
        expr = re.sub(r'\\cap', '∩', expr)  # intersection
        expr = re.sub(r'\\subset', '⊂', expr)  # subset
        expr = re.sub(r'\\supset', '⊃', expr)  # superset
        expr = re.sub(r'\\in', '∈', expr)  # element of
        expr = re.sub(r'\\notin', '∉', expr)  # not element of
        expr = re.sub(r'\\emptyset', '∅', expr)  # empty set
        expr = re.sub(r'\\forall', '∀', expr)  # for all
        expr = re.sub(r'\\exists', '∃', expr)  # exists
        expr = re.sub(r'\\neg', '¬', expr)  # negation
        expr = re.sub(r'\\wedge', '∧', expr)  # and
        expr = re.sub(r'\\vee', '∨', expr)  # or
        expr = re.sub(r'\\leq', '≤', expr)  # less than or equal
        expr = re.sub(r'\\geq', '≥', expr)  # greater than or equal
        expr = re.sub(r'\\neq', '≠', expr)  # not equal
        expr = re.sub(r'\\approx', '≈', expr)  # approximately equal
        expr = re.sub(r'\\cdot', '·', expr)  # dot multiplication
        expr = re.sub(r'\\times', '×', expr)  # multiplication
        expr = re.sub(r'\\div', '÷', expr)  # division
        expr = re.sub(r'\\pm', '±', expr)  # plus minus
        expr = re.sub(r'\\infty', '∞', expr)  # infinity
        expr = re.sub(r'\\sum', '∑', expr)  # summation
        expr = re.sub(r'\\prod', '∏', expr)  # product
        expr = re.sub(r'\\rightarrow', '→', expr)  # right arrow
        expr = re.sub(r'\\leftarrow', '←', expr)  # left arrow
        expr = re.sub(r'\\leftrightarrow', '↔', expr)  # double arrow
        # Set notation specific replacements
        expr = re.sub(r'\|([^|]+)\|', 'size(\\1)', expr)  # set size notation
        return expr

    # Process inline LaTeX
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f':latex:`{replace_math_symbols(m.group(1))}`', text)
    
    # Process multiline LaTeX blocks
    lines = text.split('\n')
    formatted_lines = []
    in_latex_block = False
    latex_content = []
    
    for line in lines:
        if line.strip() == '$$' and not in_latex_block:
            in_latex_block = True
            continue
        elif line.strip() == '$$' and in_latex_block:
            in_latex_block = False
            latex_str = '\n'.join(latex_content)
            formatted_lines.append(f':latex:`{replace_math_symbols(latex_str)}`')
            latex_content = []
            continue
            
        if in_latex_block:
            latex_content.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def process_math_expressions(text):
    """
    Process mathematical expressions to make them more readable
    """
    # Replace set operations with natural language
    text = re.sub(r'size\(([^)]+)\)', lambda m: f"the size of {m.group(1)}", text)
    text = re.sub(r'([A-Za-z])\s*∪\s*([A-Za-z])', r'\1 union \2', text)
    text = re.sub(r'([A-Za-z])\s*∩\s*([A-Za-z])', r'\1 intersect \2', text)
    text = re.sub(r'([A-Za-z])\s*⊂\s*([A-Za-z])', r'\1 is a subset of \2', text)
    text = re.sub(r'([A-Za-z])\s*⊃\s*([A-Za-z])', r'\1 is a superset of \2', text)
    text = re.sub(r'([A-Za-z])\s*∈\s*([A-Za-z])', r'\1 is an element of \2', text)
    text = re.sub(r'([A-Za-z])\s*∉\s*([A-Za-z])', r'\1 is not an element of \2', text)
    
    return text

def format_streaming_response(text, is_reasoning_prompt=True):
    """
    Formats the response for streaming, handling partial tags and incomplete content
    """
    if not is_reasoning_prompt:
        formatted_text = format_latex(text)
        return process_math_expressions(formatted_text) + "▌"
        
    # Initialize formatted text
    formatted_text = ""
    current_text = text
    
    # Check if thought section has started
    if "<|begin_of_thought|>" in current_text:
        formatted_text += "### Nova Lite Reasoning for you - I am not perfect, but will try my best🫡\n\n"
        # Split at the beginning of thought
        parts = current_text.split("<|begin_of_thought|>", 1)
        if len(parts) > 1:
            thought_content = parts[1]
            # Check if thought section has ended
            if "<|end_of_thought|>" in thought_content:
                # Get content before end tag
                thought_content = thought_content.split("<|end_of_thought|>")[0]
                current_text = current_text.split("<|end_of_thought|>", 1)[-1]
            formatted_text += format_latex(thought_content) + "\n\n"
            formatted_text = process_math_expressions(formatted_text)
    
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
            formatted_text += format_latex(solution_content)
            formatted_text = process_math_expressions(formatted_text)
    
    # If no sections have started yet, just return the text
    if formatted_text == "":
        formatted_text = format_latex(current_text)
        return process_math_expressions(formatted_text) + "▌"
    
    return formatted_text.strip() + "▌"

def format_final_response(text, is_reasoning_prompt=True, is_stream_complete=False):
    """
    Formats the final complete response, removing all tags
    """
    if not is_reasoning_prompt:
        formatted_text = format_latex(text)
        return process_math_expressions(formatted_text)
        
    # Extract thought content
    thought_match = re.search(r'<\|begin_of_thought\|>(.*?)(?:<\|end_of_thought\|>|$)', text, re.DOTALL)
    thought_content = thought_match.group(1).strip() if thought_match else ""

    # Extract solution content
    solution_match = re.search(r'<\|begin_of_solution\|>(.*?)(?:<\|end_of_solution\|>|$)', text, re.DOTALL)
    solution_content = solution_match.group(1).strip() if solution_match else ""

    # Format with headers
    formatted_text = ""
    if thought_content:
        formatted_thought = format_latex(thought_content)
        formatted_thought = process_math_expressions(formatted_thought)
        formatted_text += "### Nova Lite Reasoning for you - I am not perfect, but will try my best🫡\n\n" + formatted_thought + "\n\n"
    if solution_content:
        formatted_solution = format_latex(solution_content)
        formatted_solution = process_math_expressions(formatted_solution)
        formatted_text += "### Solution\n\n" + formatted_solution

    # Only check for missing tags if streaming is complete
    if is_reasoning_prompt and is_stream_complete and check_missing_tags(text):
        formatted_text += "\n\n⚠️ *Note: Sorry I could not complete my thought process😕. Out of Bedrock tokens. Some sections may be incomplete.*"

    if not formatted_text:
        formatted_text = format_latex(text)
        formatted_text = process_math_expressions(formatted_text)
    
    return formatted_text.strip()
    
def stream_response(client, model_id, messages, system_prompt, inference_config, additional_model_request_fields):
    """
    Streams the response from the model
    """
    system_prompts = [{"text": system_prompt}]
    
    try:
        response = client.converse_stream(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_request_fields
        )
        
        stream = response.get('stream')
        if stream:
            full_response = ""
            
            for event in stream:
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']['delta']['text']
                    full_response += chunk
                    yield chunk
                    
            return full_response
                    
    except Exception as e:
        st.error(f"An error occurred during streaming: {str(e)}")
        return None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection dropdown
    selected_model_label = st.selectbox(
        "Select Model",
        options=list(MODEL_OPTIONS.keys()),
        index=3  # Default to "Nova Lite Reasoning"
    )
    MODEL_ID = MODEL_OPTIONS[selected_model_label]
    
    # Track the selected model in session state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = MODEL_ID
    
    # Reset chat history if the model selection changes
    if st.session_state.selected_model != MODEL_ID:
        st.session_state.selected_model = MODEL_ID
        st.session_state.messages = []  # Clear chat history
        st.warning("Model changed. Chat history cleared.")  # Notify user
    
    # System prompt selection
    st.subheader("System Prompt")
    prompt_choice = st.radio(
        "Choose Prompt",
        options=["Reasoning Prompt", "Default Prompt"],
        index=0  # Default to "Reasoning Prompt"
    )
    
    # Define system prompts
    reasoning_prompt = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"""
    default_prompt = "You are a helpful agent."
    
    # Set the system prompt based on user selection
    system_prompt = reasoning_prompt if prompt_choice == "Reasoning Prompt" else default_prompt
    
    # Track the selected system prompt in session state
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = system_prompt
    
    # Reset chat history if the system prompt changes
    if st.session_state.system_prompt != system_prompt:
        st.session_state.system_prompt = system_prompt
        st.session_state.messages = []  # Clear chat history
        st.warning("System prompt changed. Chat history cleared.")  # Notify user
    
    # Display the selected system prompt in a text area
    system_prompt = st.text_area(
        "System Prompt",
        value=system_prompt,
        help="Define the AI's behavior and personality."
    )
    
    # Inference parameters
    st.subheader("Inference Parameters")
    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=32000, value=5120)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.6)
    top_k = st.number_input("Top K", min_value=1, max_value=100, value=10)

# Display the selected model ID
st.caption(f"🚀 Powered by | Model: {MODEL_ID}")

config = Config(
    read_timeout=10000,
    connect_timeout=600,
    retries={"max_attempts": 3}
)

# Initialize Bedrock client using Streamlit secrets
try:
    client = boto3.client(
        "bedrock-runtime",
        config=config,
        region_name=st.secrets.AWS["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets.AWS["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets.AWS["AWS_SECRET_ACCESS_KEY"]
    )
except Exception as e:
    st.error(f"Failed to initialize Bedrock client: {str(e)}")
    st.stop()

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the messages for streaming
    messages = [
        {"role": "user", "content": [{"text": prompt}]},
    ]
    
    inference_config = {
        "maxTokens": max_tokens,
        "topP": top_p,
        "temperature": temperature
    }
    
    additional_model_request_fields = {
        "inferenceConfig": {
            "topK": top_k
        }
    }

    # Check if reasoning prompt is selected
    is_reasoning_prompt = (prompt_choice == "Reasoning Prompt")

    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in stream_response(
            client,
            MODEL_ID,
            messages,
            system_prompt,
            inference_config,
            additional_model_request_fields
        ):
            if chunk:
                full_response += chunk
                # Format the partial response for streaming (without completion check)
                formatted_response = format_streaming_response(full_response, is_reasoning_prompt)
                response_placeholder.markdown(formatted_response)
        
        # After streaming is complete, format the final response with completion check
        formatted_final_response = format_final_response(full_response, is_reasoning_prompt, is_stream_complete=True)
        response_placeholder.markdown(formatted_final_response)
        
        # Add the complete response to chat history
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": formatted_final_response})
