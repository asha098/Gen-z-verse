from transformers import AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer
import torch

def extract_style_vector(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    style_vector = torch.mean(embeddings, dim=1)
    return style_vector

def translate_text(text, model, tokenizer):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]

def validate_style(original_text, translated_text, original_style_vector, model, tokenizer):
    translated_style_vector = extract_style_vector(translated_text, model, tokenizer)
    similarity = torch.nn.functional.cosine_similarity(original_style_vector, translated_style_vector)
    return similarity.item()

# Load models and tokenizers



# Example text


# Extract style vector

# Translate text


# Validate style transfer
