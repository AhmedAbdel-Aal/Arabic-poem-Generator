import json
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from charLSTM.model import ArabicPoetryLSTM


def load_json_as_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: '{file_path}' is not a valid JSON file.")
        return None
    except Exception as e:
        st.error(f"An error occurred while reading the file: {str(e)}")
        return None

# Define the LSTM model class (same as before)

# Load the trained model and necessary data
@st.cache_resource
def load_model():
    # Load vocabulary
    # Note: You need to save and load your vocabulary along with the model
    # This is a placeholder. Replace with your actual vocabulary loading code
    idx_to_char = load_json_as_dict('./charLSTM/charLSTM_idx_to_char.json')
    char_to_idx = load_json_as_dict('./charLSTM/charLSTM_char_to_idx.json')

    
    # Model parameters
    vocab_size = len(idx_to_char.keys())
    hidden_dim = 512  # Should match your training configuration
    num_layers = 2    # Should match your training configuration
    
    # Initialize the model
    model = ArabicPoetryLSTM(vocab_size, hidden_dim, num_layers)
    
    # Load the trained weights
    model.load_state_dict(torch.load('./charLSTM/char_LSTM.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return model, char_to_idx, idx_to_char


def top_k_sampling(logits, k=5, temperature=1.0):
    top_k = min(k, logits.size(-1))  # Safety check
    # Apply temperature
    logits = logits / temperature
    # Get the top k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
    # Apply softmax to convert to probabilities
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    # Sample from the top k probabilities
    sampled_index = torch.multinomial(top_k_probs, num_samples=1)
    # Get the actual character index
    char_index = top_k_indices[sampled_index].item()
    return char_index

# Text generation function
def generate_text(start_sequence: str, max_length: int = 100, temperature: float = 1.0) -> str:
    device = torch.device('cpu')  # Use GPU if available
    model.to(device)
    
    current_sequence = start_sequence
    generated_sequence = start_sequence
    
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        for _ in range(max_length):
            input_seq = torch.tensor([char_to_idx.get(ch, char_to_idx['<pad>']) for ch in current_sequence]).unsqueeze(0).to(device)
            output, hidden = model(input_seq, hidden)
            
            last_char_logits = output[0, -1, :] / temperature
            #last_char_probs = F.softmax(last_char_logits, dim=0)
            last_char_index = top_k_sampling(last_char_logits, k=10, temperature=temperature)
            #torch.multinomial(last_char_probs, 1).item()
            
            predicted_char = idx_to_char[str(last_char_index)]
            generated_sequence += predicted_char
            
            current_sequence = current_sequence[1:] + predicted_char
            
            if predicted_char == '-':  # End of line character
                break
    
    return generated_sequence


model, char_to_idx, idx_to_char = load_model()

# Streamlit app
st.title("Arabic Poetry Generator")

start_sequence = st.text_input("Enter the starting sequence:")
max_length = st.slider("Maximum length of generated text:", 10, 500, 100)
temperature = st.slider("Temperature (creativity):", 0.1, 2.0, 1.0, 0.1)

if st.button("Generate Poetry"):
    if start_sequence:
        with st.spinner("Generating poetry..."):
            generated_text = generate_text(start_sequence, max_length, temperature)
        st.success("Poetry generated successfully!")
        st.text_area("Generated Poetry:", value=generated_text, height=200)
    else:
        st.error("Please enter a starting sequence.")

st.markdown("---")
st.markdown("This app uses a character-level LSTM model trained on Arabic poetry to generate new poetic text.")