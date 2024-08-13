from flask import Flask, render_template, request
from transalatemodel import extract_style_vector, translate_text, validate_style
from transformers import AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer
import PyPDF2
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def translator():
    input_text = ""
    translated_text = ""
    
    if request.method == 'POST':
        if 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            if pdf_file.filename != '':
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(pdf_path)
                          
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    extracted_text = ""
                    for page in reader.pages:
                        extracted_text += page.extract_text()
                
                return render_template('translator.html', translated_text=extracted_text, input_text=input_text)

        if 'input_text' in request.form and not input_text:
            input_text = request.form.get('input_text')
        
        translated_text = input_text  
        fromlang = request.form.get('from_language')
        tolang = request.form.get('to_language')
        style_model_name = "bert-base-uncased"
        translation_model_name = language(fromlang,tolang)
        style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
        style_model = AutoModel.from_pretrained(style_model_name)
        translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        translation_model = MarianMTModel.from_pretrained(translation_model_name)
        original_text = input_text
        original_style_vector = extract_style_vector(original_text, style_model, style_tokenizer)
        translated_text = translate_text(original_text, translation_model, translation_tokenizer)
        similarity_score = validate_style(original_text, translated_text, original_style_vector, style_model, style_tokenizer)

    return render_template('translator.html', translated_text=translated_text, input_text=input_text)
def language(fromlang,tolang):
    if request.method == 'POST':
        # fromlang = request.form.get('from_language')
        # tolang = request.form.get('to_language')

        # Dictionary to map language pairs to model names
        model_mapping = {
            ("English", "Spanish"): "Helsinki-NLP/opus-mt-en-es",
            ("Spanish", "English"): "Helsinki-NLP/opus-mt-es-en",
            ("French", "English"): "Helsinki-NLP/opus-mt-fr-en",
            ("English", "French"): "Helsinki-NLP/opus-mt-en-fr",
            ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
            ("English", "Italian"): "Helsinki-NLP/opus-mt-en-it",
          #  ("English", "Portuguese"): "Helsinki-NLP/opus-mt-en-pt",
            ("English", "Romanian"): "Helsinki-NLP/opus-mt-en-ro",
            ("English", "Russian"): "Helsinki-NLP/opus-mt-en-ru",
            ("English","Swedish"): "Helsinki-NLP/opus-mt-en-sv",
            # ("English", "English"): "Helsinki-NLP/opus-mt-en-en",
             #("Spanish", "Spanish"): "Helsinki-NLP/opus-mt-es-es",
             # ("French", "French"): "Helsinki-NLP/opus-mt-fr-fr",
            # Add more language pairs here
        }

        # Look up the model name based on the language pair
        model_name = model_mapping.get((fromlang, tolang))

        if model_name:
            return model_name
        else:
            return "Translation model not available for the requested language pair."
    
    return "Invalid request method. Please use POST."

if __name__ == '__main__':
    app.run(debug=True)
