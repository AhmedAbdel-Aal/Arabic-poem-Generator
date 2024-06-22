import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    step = 0
    for inputs, targets, mask in dataloader:
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        if step == 0:
          hidden = model.init_hidden(inputs.size(0), device)
        
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)

        loss = criterion(output.transpose(1, 2), targets)
        loss = (loss * mask).sum() / mask.sum()  # Average loss over non-padded elements
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, mask in dataloader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            hidden = model.init_hidden(inputs.size(0), device)
            
            output, _ = model(inputs, hidden)
            loss = criterion(output.transpose(1, 2), targets)
            loss = (loss * mask).sum() / mask.sum()  # Average loss over non-padded elements
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_text(model, dataset, start_sequence, max_length=100, temperature=1.0, device='cpu'):
    model.eval()
    current_sequence = start_sequence
    generated_sequence = start_sequence
    
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        for _ in range(max_length):
            input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in current_sequence]).unsqueeze(0).to(device)
            output, hidden = model(input_seq, hidden)
            
            # Get the last character's prediction
            last_char_logits = output[0, -1, :] / temperature
            last_char_probs = F.softmax(last_char_logits, dim=0)
            last_char_index = torch.multinomial(last_char_probs, 1).item()
            
            predicted_char = dataset.idx_to_char[last_char_index]
            generated_sequence += predicted_char
            
            # Update current_sequence for next iteration
            current_sequence = current_sequence[1:] + predicted_char
            
            if predicted_char == '-':  # End of line character
                break
    
    return generated_sequence

