from flask import Flask, request, jsonify, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import pandas as pd

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# Rota principal - página HTML
@app.route('/')
def index():
    return render_template('index.html')

# Rota para análise de texto único
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if not text.strip():
        return jsonify({"error": "Texto vazio"}), 400
    
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    # Determinar o "mood" do texto
    if compound > 0.05:
        mood = "positivo"
    elif compound < -0.05:
        mood = "negativo"
    else:
        mood = "neutro"

    return jsonify({"mood": mood, "scores": scores})

# Rota para upload de arquivo
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    try:
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            return jsonify({"error": "O arquivo deve conter uma coluna chamada 'text'"}), 400
        
        results = []
        for text in df['text']:
            scores = sia.polarity_scores(str(text))
            compound = scores['compound']
            mood = "positivo" if compound > 0.05 else "negativo" if compound < -0.05 else "neutro"
            results.append({"text": text, "mood": mood, "scores": scores})
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.secret_key = os.urandom(24)
    app.run(debug=True)