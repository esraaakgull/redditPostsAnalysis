from transformers import pipeline
import pandas as pd
import re
from textblob import TextBlob


def analyze_sentiment(sentence):
    # Creating a TextBlob object
    blob = TextBlob(sentence)

    # Getting the polarity score, which ranges from -1 (negative) to 1 (positive)
    polarity = blob.sentiment.polarity

    return polarity


# BERT tabanlı duygu analizi modelini yükleyin
sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Programlama dilleri listesi
programming_languages = [
    "ABAP", "ActionScript", "Ada", "ALGOL", "Alice", "APL", "Assembly", "AutoIt", "AutoLISP", "Bash",
    "C", "C#", "C++", "COBOL", "Clojure", "COOL", "Crystal", "Dart", "Delphi", "Eiffel", "Elixir",
    "Elm", "Erlang", "F#", "Forth", "Fortran", "Groovy", "Haskell", "HTML", "Java", "JavaScript",
    "Julia", "Kotlin", "Lisp", "Lua", "MATLAB", "Objective-C", "Pascal", "Perl", "PHP", "Prolog", "Python",
    "Ruby", "Rust", "Scala", "Scheme", "Shell", "Swift", "Tcl", "TypeScript", "VBScript", "Verilog",
    "VHDL", "Visual Basic .NET"
]

# Verileri işleyin ve sonuçları saklamak için bir liste oluşturun
results = []

df = pd.read_excel('dil_tespiti.xlsx')

# Her satırdaki metinleri işleyin
for index, row in df.iterrows():
    detected_sentences = row['Detected_Sentences']
    if not isinstance(detected_sentences, str) or pd.isna(detected_sentences):
        continue

    cümleler = detected_sentences.split('\n')  # Metin cümlelerini alın

    # Her cümle için analiz yapın ve sadece programlama dilleri içerenleri seçin
    for cümle in cümleler:
        for dil in programming_languages:
            if re.search(rf'\b{re.escape(dil)}\b', cümle, re.IGNORECASE):
                sonuç = sentiment_analysis(cümle)
                textlob_polarity = analyze_sentiment(cümle)
                results.append({
                    'ID': row['ID'],
                    'Sentiment Analizi Gerçekleştirilen Dil': dil,
                    'Cümle': cümle,
                    'Sentiment Analizi Sonucu': sonuç[0]['label'],
                    'TextBlob Polarity Sonucu': textlob_polarity,
                    'Score': sonuç[0]['score']
                })

# Sonuçları yeni bir DataFrame'e dönüştürün ve Excel dosyasına kaydedin
analiz_df = pd.DataFrame(results)
analiz_df.to_excel('sentiment_analysis_results2.xlsx', index=False)
