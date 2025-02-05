import json
import boto3
import streamlit as st
from botocore.config import Config
import re  # Used for post-processing math expressions

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Amazon Bedrock Reasoning Capability Test Platform", layout="wide")

# Inject MathJax for rendering LaTeX expressions
st.markdown(
    """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS_HTML" type="text/javascript"></script>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Amazon Bedrock Reasoning Capability Test Platform")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model configuration
    MODEL_OPTIONS = {
        "Nova Micro": "us.amazon.nova-micro-v1:0",
        "Nova Lite": "us.amazon.nova-lite-v1:0",
        "Nova Pro": "us.amazon.nova-pro-v1:0",
        "Nova Lite Reasoning": "arn:aws:bedrock:us-east-1:963067361214:provisioned-model/dx9g06qbz7f6",
    }
    
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
        st.warning("Model changed. Chat history cleared.")
    
    # System prompt selection
    st.subheader("System Prompt")
    prompt_choice = st.radio(
        "Choose Prompt",
        options=["Reasoning Prompt", "Default Prompt"],
        index=0  # Default to "Reasoning Prompt"
    )
    
    # Define system prompts
    reasoning_prompt = (
        "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. "
        "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
        "Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> "
        "Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
        "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. "
        "The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> "
        "Now, try to solve the following question through the above guidelines:"
    )
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
        st.warning("System prompt changed. Chat history cleared.")
    
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

def wrap_math_expressions(text: str) -> str:
    """
    Scans through the given text and replaces any substring enclosed in matching parentheses
    that appears to be a LaTeX math expression (heuristically determined by the presence of '\\' or '^')
    with the same content wrapped in dollar signs for proper MathJax rendering.
    """
    result = ""
    i = 0
    while i < len(text):
        if text[i] == '(':
            start = i
            depth = 0
            j = i
            # Find the matching closing parenthesis
            while j < len(text):
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            # If a matching parenthesis is found, consider it a candidate
            if j < len(text) and depth == 0:
                candidate = text[start+1:j]
                # Heuristic: if the candidate contains a LaTeX command or caret,
                # assume it is a math expression.
                if ('\\' in candidate) or ('^' in candidate):
                    # Replace outer parentheses with inline math delimiters.
                    result += "$" + candidate.strip() + "$"
                    i = j + 1
                    continue
        # If no math expression is detected or no matching parenthesis, copy the character as is.
        result += text[i]
        i += 1
    return result

def stream_response(client, model_id, messages, system_prompt, inference_config, additional_model_request_fields):
    """
    Streams the response from the model.
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
                    # Automatically wrap math expressions in the full response before rendering
                    rendered_response = wrap_math_expressions(full_response)
                    yield rendered_response
            return full_response
                    
    except Exception as e:
        st.error(f"An error occurred during streaming: {str(e)}")
        return None

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
    
    # Separate inferenceConfig and additionalModelRequestFields
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

    # Create a placeholder for the streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response and update in real-time
        for rendered_chunk in stream_response(
            client,
            MODEL_ID,
            messages,
            system_prompt,
            inference_config,
            additional_model_request_fields
        ):
            if rendered_chunk:
                # Update the response placeholder with a cursor symbol
                response_placeholder.markdown(rendered_chunk + "▌")
        
        # Final rendering without the cursor
        final_rendered = wrap_math_expressions(full_response)
        response_placeholder.markdown(final_rendered)
        
        # Add the complete response to chat history if available
        if final_rendered:
            st.session_state.messages.append({"role": "assistant", "content": final_rendered})
