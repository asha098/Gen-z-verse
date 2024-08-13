from flask import Flask, render_template, request
from transalatemodel import extract_style_vector, translate_text, validate_style
from transformers import AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer



app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def translator():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        # translated_text = input_text
        style_model_name = "bert-base-uncased"
        translation_model_name = "Helsinki-NLP/opus-mt-en-es"
        style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
        style_model = AutoModel.from_pretrained(style_model_name)
        translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        translation_model = MarianMTModel.from_pretrained(translation_model_name)
        original_text = input_text
        original_style_vector = extract_style_vector(original_text, style_model, style_tokenizer)
        translated_text = translate_text(original_text, translation_model, translation_tokenizer)
        similarity_score = validate_style(original_text, translated_text, original_style_vector, style_model, style_tokenizer)
        # print(f"Original text: {original_text}")
        # print(f"Translated text: {translated_text}")        "This is an example sentence from James Joyce's work."
        # print(f"Style similarity score: {similarity_score}")
        return render_template('translator.html', translated_text=translated_text, input_text=input_text)
    return render_template('translator.html')
if __name__ == '__main__':
    app.run(debug=True)