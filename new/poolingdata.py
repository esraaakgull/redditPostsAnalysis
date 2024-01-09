import praw
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import string
import re
import time

# nltk'yi başlatın (ilk çalıştırmada gereklidir)
# nltk.download('punkt')

# Define user agent
user_agent = "praw_scraper_1.0"

# Create an instance of reddit class
reddit = praw.Reddit(username="eesraakgull",
                     password="Esra.1998",
                     client_id="dBScuPJw95o8DFw1xLyLDw",
                     client_secret="ScPMZoWZOFA7Q7IMrEKtvBtvzUIp5w",
                     user_agent=user_agent
                     )

# Create sub-reddit instance
subreddit_name = "programmingLanguages"
subreddit = reddit.subreddit(subreddit_name)

# Verileri saklamak için boş bir liste oluşturun
data = {'ID': [],
        'Başlık': [],
        'İçerik': [],
        'Yorumlar': []}

# Subreddit'teki en popüler 10 gönderiyi alın (istediğiniz sayıyı ayarlayabilirsiniz)
for submission in subreddit.top(limit=5):
    # Gönderinin başlığı ve içeriğini alın
    baslik = submission.title
    icerik = submission.selftext

    # Metni cümlelere ayırın
    baslik_cümleler = sent_tokenize(baslik)
    icerik_cümleler = sent_tokenize(icerik)

    # Veriyi saklayın
    data['ID'].append(submission.id)
    data['Başlık'].append('\n'.join(baslik_cümleler))
    data['İçerik'].append('\n'.join(icerik_cümleler))

    # Gönderinin yorumlarını alın
    yorumlar = []
    for comment in submission.comments:
        yorum = comment.body
        yorum_cümleler = sent_tokenize(yorum)
        yorumlar.extend(yorum_cümleler)

    data['Yorumlar'].append('\n'.join(yorumlar))
    time.sleep(10)

# Verileri bir pandas DataFrame'e yükleyin
df = pd.DataFrame(data)


# Linkleri ve emojileri kaldırın
def remove_links_and_emojis(text):
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

# Gereksiz noktalama işaretlerini kaldırın , punctuation
df['Başlık'] = df['Başlık'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['İçerik'] = df['İçerik'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['Yorumlar'] = df['Yorumlar'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Veriyi Excel dosyasına kaydedin
df.to_excel('programminglanguages_allsubreddit.xlsx', index=False)
