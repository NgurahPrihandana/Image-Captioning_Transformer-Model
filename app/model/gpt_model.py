from transformers import VisionEncoderDecoderModel, ViTModel, GPT2LMHeadModel, GPT2Config, ViTFeatureExtractor, GPT2TokenizerFast
import torch

from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder and decoder base models
encoder_model = "google/vit-base-patch16-224-in21k"
decoder_model = "gpt2"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model, 
    decoder_model
)

# Now load your fine-tuned weights
model.load_state_dict(torch.load(f"{BASE_DIR}/gpt_model.pt", map_location=device))
model.to(device)
model.eval()

# Set special tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id

feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model)

# Now do inference on a local image
from PIL import Image

# def generate_caption(file_path):
#     image_path = file_path
#     image = Image.open(image_path).convert("RGB")

#     inputs = feature_extractor(images=image, return_tensors="pt").to(device)
#     generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=30, num_beams=4)
#     caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

#     return caption

def generate_caption(file_path):
    image_path = file_path
    image = Image.open(image_path).convert("RGB")
    
    # Explicitly resize the image to 224x224 pixels
    desired_size = (224, 224)
    image = image.resize(desired_size, resample=Image.BILINEAR)

    # Preprocess the image using the feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    
    # Generate caption using the model
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"], 
        max_length=30, 
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the generated IDs to obtain the caption
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return caption


file_name = f"{BASE_DIR}/images/test1.jpg"
captions = generate_caption(file_name)
print(captions)

