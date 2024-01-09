import pandas as pd

# Temizlenmiş verilerin olduğu Excel dosyasını oku
input_file_path = 'temizlenmis_veriler.xlsx'  # Temizlenmiş verilerin Excel dosyasının yolu
df = pd.read_excel(input_file_path)

# Programlama dillerini içeren liste
programming_languages = [
    "abap",
    "actionscript",
    "ada",
    "algol",
    "alice",
    "apl",
    "assembly",
    "autoit",
    "autolisp",
    "bash",
    "c",
    "c#",
    "c++",
    "cobol",
    "clojure",
    "cool",
    "crystal",
    "d",
    "dart",
    "delphi",
    "eiffel",
    "elixir",
    "elm",
    "erlang",
    "f#",
    "forth",
    "fortran",
    "go",
    "groovy",
    "haskell",
    "html",
    "java",
    "javascript",
    "julia",
    "kotlin",
    "lisp",
    "lua",
    "matlab",
    "objective-c",
    "pascal",
    "perl",
    "php",
    "prolog",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "scheme",
    "shell",
    "swift",
    "tcl",
    "typescript",
    "vbscript",
    "verilog",
    "vhdl",
    "visual basic"
    ".net"
]

# Cümleleri ve programlama dillerini belirleme
detected_languages = []
detected_sentences = []

for index, row in df.iterrows():
    # Tüm metinleri string olarak birleştirin
    baslik = str(row['Başlık'])
    icerik = str(row['İçerik'])
    yorumlar = str(row['Yorumlar'])

    cümle = baslik + ' ' + icerik + ' ' + yorumlar

    detected_language = []
    detected_sentence = []

    for language in programming_languages:
        if language.lower() in cümle.lower():
            detected_language.append(language)
            detected_sentence.append(cümle)  # Eşleşen cümleyi ekleyin

    detected_languages.append(', '.join(detected_language))
    detected_sentences.append('\n'.join(detected_sentence))

# Verilere yeni iki sütun ekleyin: "Detected_Languages" ve "Detected_Sentences"
df['Detected_Languages'] = detected_languages
df['Detected_Sentences'] = detected_sentences

# Yalnızca eşleşen cümleleri içeren sütunu temizleyin
df['Detected_Sentences'] = df['Detected_Sentences'].apply(lambda x: '' if x == '\n'.join([]) else x)

# Sonuçları yeni bir Excel dosyasına kaydedin
output_file_path = 'dil_tespiti.xlsx'  # Dil tespit sonuçlarının kaydedileceği Excel dosyasının yolu
df.to_excel(output_file_path, index=False)
