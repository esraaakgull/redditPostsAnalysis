import pandas as pd
import string
import re

# Excel dosyasını oku
input_file_path = 'programminglanguages_allsubreddit.xlsx'  # Verilerinizin Excel dosyasının yolu
df = pd.read_excel(input_file_path)


# Linkleri ve emojileri kaldırın
def remove_links_and_emojis(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Linkleri kaldır
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emojiler (duygusal ifadeler)
                                   u"\U0001F300-\U0001F5FF"  # semboller ve işaretler
                                   u"\U0001F680-\U0001F6FF"  # taşıtlar ve simgeler
                                   u"\U0001F700-\U0001F77F"  # alaka işaretleri
                                   u"\U0001F780-\U0001F7FF"  # geometrik şekiller
                                   u"\U0001F800-\U0001F8FF"  # çin astrolojisi
                                   u"\U0001F900-\U0001F9FF"  # kuaför
                                   u"\U0001FA00-\U0001FA6F"  # sporlar
                                   u"\U0001FA70-\U0001FAFF"  # yiyecek ve içecek
                                   u"\U0001FB00-\U0001FBFF"  # kahramanlar
                                   u"\U0001F004-\U0001F0CF"  # kart oyunları
                                   u"\U0001F170-\U0001F251"  # semboller (diğer)
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)  # Emoji'leri kaldır
    return text


# Verilerdeki linkleri, emojileri ve diğer alfabeleri kaldırın
df['Başlık'] = df['Başlık'].apply(remove_links_and_emojis)
df['İçerik'] = df['İçerik'].apply(remove_links_and_emojis)
df['Yorumlar'] = df['Yorumlar'].apply(remove_links_and_emojis)

# Gereksiz noktalama işaretlerini kaldırın
df['Başlık'] = df['Başlık'].apply(
    lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)
df['İçerik'] = df['İçerik'].apply(
    lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)
df['Yorumlar'] = df['Yorumlar'].apply(
    lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

# Temizlenmiş verileri farklı bir Excel dosyasına kaydet
output_file_path = 'temizlenmis_veriler.xlsx'  # Temizlenmiş verilerin kaydedileceği Excel dosyasının yolu
df.to_excel(output_file_path, index=False)
