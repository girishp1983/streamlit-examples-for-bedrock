import json
import boto3
import streamlit as st

# App configuration
st.set_page_config(page_title="Amazon Bedrock Chat", layout="wide")
st.title("💬 Amazon Bedrock Chat")
st.caption("🚀 Powered by Nova via Amazon Bedrock Converse API")

# Initialize Bedrock client using Streamlit secrets
try:
    client = boto3.client(
        "bedrock-runtime",
        region_name=st.secrets.AWS["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets.AWS["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets.AWS["AWS_SECRET_ACCESS_KEY"]
    )
except Exception as e:
    st.error(f"Failed to initialize Bedrock client: {str(e)}")
    st.stop()

# Model configuration
MODEL_ID = "us.amazon.nova-lite-v1:0"

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # System prompt input
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        help="Define the AI's behavior and personality."
    )
    
    # Inference parameters
    st.subheader("Inference Parameters")
    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=300)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3)
    top_k = st.number_input("Top K", min_value=1, max_value=100, value=20)

# Chat interface
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

    # Prepare the request body for Converse API
    messages = [
        {"role": "user", "content": [{"text": prompt}]},
    ]
    
    system = [{"text": system_prompt}]
    
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

    # Call the Converse API
    try:
        model_response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            system=system,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_request_fields
        )
        
        # Extract and display the response
        response_text = model_response["output"]["message"]["content"][0]["text"]
        
        with st.chat_message("assistant"):
            st.markdown(response_text)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
