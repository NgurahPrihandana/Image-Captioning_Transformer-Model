# Step 1: Import Libraries
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

from torch.nn.utils.rnn import pad_sequence


import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.distributions import Categorical

# torch.backends.cuda.matmul.allow_tf32 = True

import nltk
from nltk.corpus import wordnet

from transformers import ViTModel, ViTFeatureExtractor
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Memuat pre-trained model dan feature extractor
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
# Define a decoder module for the Transformer architecture
class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=768, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()

        # Create an embedding layer for tokens
        self.embedding = nn.Embedding(num_emb, hidden_size)

        # Positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=0.0,
            batch_first=True
        )
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq, encoder_output, input_padding_mask=None, encoder_padding_mask=None):
        # Embedding and positional embeddings
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape
        pos_emb = self.pos_emb(torch.arange(l, device=input_seq.device)).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb
    
        # Handle optional padding mask
        #if input_padding_mask is not None:
            #print("Padding Mask Shape:", input_padding_mask.shape)
            #print("Padding Mask (Sample):", input_padding_mask[0])
    
        # Causal mask
        causal_mask = torch.triu(torch.ones(l, l, device=input_seq.device), 1).bool()
        #print("Causal Mask Shape:", causal_mask.shape)
    
        # Pass through transformer decoder layers
        output = self.decoder_layers(
            tgt=embs, memory=encoder_output, tgt_mask=causal_mask,
            tgt_key_padding_mask=input_padding_mask, memory_key_padding_mask=encoder_padding_mask
        )
        return self.fc_out(output)
    
# Define an Vision Encoder-Decoder module for the Transformer architecture
class VisionEncoderDecoder(nn.Module):
  
    def __init__(self, encoder, decoder):
        super(VisionEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_image, target_seq, padding_mask):
        # Use input_image directly (already preprocessed in the training loop)
        encoder_outputs = self.encoder(**input_image).last_hidden_state
    
        # Ensure padding_mask is bool
        padding_mask = padding_mask.bool()
    
        # Decode using the decoder
        decoded_seq = self.decoder(input_seq=target_seq, encoder_output=encoder_outputs,
                                   input_padding_mask=padding_mask)
        return decoded_seq
    
# Embedding Size
hidden_size = 768

# Number of Transformer blocks for the (Encoder, Decoder)
num_layers = (6, 6)

# MultiheadAttention Heads
num_heads = 8

# Size of the patches
patch_size = 8

# Load the tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Replace if custom tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# ===== 1. Load the Model ===== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
caption_model = VisionEncoderDecoder(
    encoder=vit_model,  # Ganti vit_model menjadi encoder
    decoder=Decoder(num_emb=tokenizer.vocab_size, hidden_size=hidden_size,
                    num_layers=num_layers[1], num_heads=num_heads)
).to(device)

caption_model.load_state_dict(torch.load(f"{BASE_DIR}/best_model.pt", map_location=device))
caption_model.eval()

def generate_caption(file_name, max_length=30, temp=0.7):
    """
    Generate a caption for an image file using the trained model.
    Args:
        file_name (str): Path to the image file.
        max_length (int): Maximum length of the caption.
        temp (float): Sampling temperature.
    Returns:
        str: Generated caption.
    """
    # Check if the file exists
    assert os.path.exists(file_name), f"Image path does not exist: {file_name}"

    # Load and preprocess the image with resizing
    image = Image.open(file_name).convert("RGB")
    image = image.resize((128, 128))  # Resize to 128x128
    image_tensor = feature_extractor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # Start of Sentence (SOS) token
    sos_token = torch.tensor([[tokenizer.cls_token_id]]).to(device)  # Start token
    tokens = [sos_token]

    with torch.no_grad():
        # Get image features using the encoder
        image_embedding = caption_model.encoder(pixel_values=image_tensor).last_hidden_state

        for _ in range(max_length):
            input_tokens = torch.cat(tokens, dim=1)  # Concatenate tokens
            outputs = caption_model.decoder(input_seq=input_tokens, encoder_output=image_embedding)

            # Sample the next token with temperature
            logits = outputs[:, -1] / temp  # Use logits of last token
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

            # Append next token and stop if end token is generated
            tokens.append(next_token)

            if next_token.item() == tokenizer.sep_token_id:  # End token
                break

    # Decode generated tokens
    caption = tokenizer.decode(torch.cat(tokens, dim=1)[0], skip_special_tokens=True)
    return caption

# Example usage
# file_name = f"{BASE_DIR}/images/test2.jpg"
# caption = generate_caption(file_name)
