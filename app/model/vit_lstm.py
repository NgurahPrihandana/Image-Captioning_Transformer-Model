import torch
import torch.nn as nn
from transformers import ViTModel, DistilBertTokenizer, DistilBertModel
from PIL import Image
import torchvision.transforms as transforms
from nltk.tokenize import word_tokenize
import nltk

from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Define preprocess_text function
def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing and tokenizing.

    Args:
        text (str): The input string.

    Returns:
        tuple: A tuple containing the list of tokens and the reconstructed string.
    """
    tokens = word_tokenize(text.lower())
    reconstructed_text = ' '.join(tokens)
    return tokens, reconstructed_text

# Define the TokenDrop class (used only during training, can be omitted during inference)
class TokenDrop(nn.Module):
    """Randomly replace tokens with the pad token during training.

    Args:
        prob (float): probability of replacing a token
        pad_token_id (int): index for the padding token
        eos_token_id (int): index for the end-of-sentence token
    """

    def __init__(self, prob=0.2, pad_token_id=0, eos_token_id=2):
        super(TokenDrop, self).__init__()
        self.prob = prob
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def forward(self, input_ids):
        if not self.training:
            return input_ids

        # Generate mask with probability 'prob'
        mask = torch.bernoulli(torch.full(input_ids.shape, self.prob)).to(input_ids.device)
        mask = mask.bool()

        # Do not replace special tokens
        mask = mask & (input_ids != self.eos_token_id) & (input_ids != self.pad_token_id)

        # Replace selected tokens with pad_token_id
        input_ids[mask] = self.pad_token_id

        return input_ids

# Define the Encoder-Decoder Model with LSTM
class ViT_LSTM_CaptioningModel(nn.Module):
    def __init__(self, encoder, embedding_dim, hidden_dim, vocab_size, pad_token_id, tokenizer, num_layers=1, dropout=0.5):
        super(ViT_LSTM_CaptioningModel, self).__init__()
        self.encoder = encoder  # ViT encoder

        # DistilBERT Embedding
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.embedding = self.distilbert.embeddings.word_embeddings  # (vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.token_drop = TokenDrop(prob=0.2, pad_token_id=pad_token_id, eos_token_id=tokenizer.sep_token_id)

        # Linear layers to map image features to LSTM hidden and cell states
        self.hidden_init = nn.Linear(encoder.config.hidden_size, hidden_dim)
        self.cell_init = nn.Linear(encoder.config.hidden_size, hidden_dim)

    def forward(self, images, input_ids, attention_mask, hidden=None):
        # Encode images
        encoder_outputs = self.encoder(images).last_hidden_state  # (batch_size, seq_length, encoder_dim)

        # Extract image features (e.g., using the [CLS] token)
        img_features = encoder_outputs[:, 0, :]  # (batch_size, encoder_dim)

        # Initialize hidden and cell states from image features
        if hidden is None:
            h_0 = self.hidden_init(img_features).unsqueeze(0)  # (num_layers, batch, hidden_dim)
            c_0 = self.cell_init(img_features).unsqueeze(0)    # (num_layers, batch, hidden_dim)
        else:
            h_0, c_0 = hidden

        # Apply token drop
        input_ids = self.token_drop(input_ids)

        # Get embeddings
        embeddings = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)

        # Pass embeddings through LSTM
        outputs, hidden = self.lstm(embeddings, (h_0, c_0))  # (batch_size, seq_length, hidden_dim)

        # Predict next tokens
        logits = self.fc(outputs)  # (batch_size, seq_length, vocab_size)

        return logits, hidden

    def generate_caption(self, images, max_length=50, device='cuda'):
        self.eval()

        # Encode images
        with torch.no_grad():
            encoder_outputs = self.encoder(images).last_hidden_state  # (batch_size, seq_length, encoder_dim)
            img_features = encoder_outputs[:, 0, :]  # (batch_size, encoder_dim)
            h_0 = self.hidden_init(img_features).unsqueeze(0)  # (num_layers, batch, hidden_dim)
            c_0 = self.cell_init(img_features).unsqueeze(0)    # (num_layers, batch, hidden_dim)

        # Initialize generated_ids with the start token
        # Adjust the start token based on your tokenizer's configuration
        generated_ids = torch.zeros((images.size(0), 1), dtype=torch.long).fill_(tokenizer.cls_token_id).to(device)

        hidden = (h_0, c_0)

        for _ in range(max_length):
            # Embed the last generated token
            embeddings = self.embedding(generated_ids)  # (batch_size, current_length, embedding_dim)

            # Pass through LSTM
            outputs, hidden = self.lstm(embeddings, hidden)  # (batch_size, current_length, hidden_dim)

            # Get logits for the last time step
            logits = self.fc(outputs[:, -1, :])  # (batch_size, vocab_size)

            # Predict the next token
            predicted = logits.argmax(dim=-1).unsqueeze(1)  # (batch_size, 1)

            # Append predicted token to generated_ids
            generated_ids = torch.cat((generated_ids, predicted), dim=1)  # (batch_size, current_length +1)

            # Check if all sequences have generated EOS token
            if (predicted == tokenizer.sep_token_id).all():
                break

        return generated_ids

# Initialize Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize ViT Encoder
vit_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def generate_caption(image_path, max_length=50):
    """
    Generates a caption for the given image using the trained model.

    Args:
        image_path (str): Path to the input image.
        max_length (int): Maximum length of the generated caption.

    Returns:
        str: Generated caption.
    """
    # Initialize the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Initialize ViT Encoder
    vit_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Define model parameters
    embedding_dim = vit_encoder.config.hidden_size  # Typically 768 for ViT
    hidden_dim = 512  # Adjust based on your training
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id

    # Instantiate the model
    model = ViT_LSTM_CaptioningModel(
        encoder=vit_encoder,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        tokenizer=tokenizer,
        num_layers=1,
        dropout=0.5
    )

    # Load the saved model weights
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    model.load_state_dict(torch.load(f"{BASE_DIR}/final_model_lstm.pt", map_location='cpu'))
    model.eval()

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Values used for ViT
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate_caption(
            image,
            max_length=max_length,
            device=device
        )
    
    # Decode the generated tokens to string
    pred_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return pred_caption

# Usage
# image_path = f"{BASE_DIR}/images/test4.jpg"
# generated_caption = generate_caption(image_path, max_length=50)
# print(generated_caption)