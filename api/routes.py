from flask import Blueprint, request, jsonify
from model.model import ArabicPoetryLSTM
import torch

api_bp = Blueprint('api', __name__)

# Load the trained model (you'll need to implement model loading)
model = ArabicPoetryLSTM(...)  # Initialize with correct parameters
model.load_state_dict(torch.load('saved_models/model.pth'))
model.eval()

@api_bp.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    seed_text = data.get('seed_text', '')
    max_length = data.get('max_length', 200)
    
    generated_text = model.generate(seed_text, max_length)
    
    return jsonify({'generated_text': generated_text})