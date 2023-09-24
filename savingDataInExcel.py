import praw
from textblob import TextBlob
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from transformers import pipeline
import umap

# There are 2 different result arrays
# result_textblob array for textBlob
# result_bert array for BERT

# languages
languages = [
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

# posts
posts = []

# {"C":{negative:[], weaklyNegative:[], neutral:[], weaklyPositive:[], positive:[]}, "Python":{negative:[], weaklyNegative:[], neutral:[], weaklyPositive:[], positive:[]},}
result_textblob = {}
result_bert = {}

filename = "newAllData.xlsx"

excel_data = []

histogram_textblob = []  # for polarity values for each language
histogram_bert = []  # for polarity values for each language

online_sentiment_analyze = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


# pca with total number of negative, weaklyNegative, neutral, weaklyPositive, positive

def pca():
    # Perform PCA for TextBlob data
    pca_textblob = PCA(n_components=1)  # Set n_components to 1 or remove it to keep all components
    textblob = np.array(histogram_textblob).reshape(-1, 1)  # Flatten and reshape for PCA
    histogram_textblob_pca = pca_textblob.fit_transform(textblob)

    # Perform PCA for BERT data
    pca_bert = PCA(n_components=1)  # Set n_components to 1 or remove it to keep all components
    bert = np.array(histogram_bert).reshape(-1, 1)  # Flatten and reshape for PCA
    histogram_bert_pca = pca_bert.fit_transform(bert)

    # Assuming you have a list of languages associated with the data
    languages_textblob = []  # List to store language labels for TextBlob
    languages_bert = []  # List to store language labels for BERT

    for i, language in enumerate(result_textblob):
        for polarity_type in ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]:
            for polarity_data in result_textblob[language][polarity_type]:
                languages_textblob.append(language)

    for i, language in enumerate(result_bert):
        for polarity_type in ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]:
            for polarity_data in result_bert[language][polarity_type]:
                languages_bert.append(language)

    # Plot the PCA results with labels for TextBlob
    plt.figure()
    plt.scatter(histogram_textblob_pca[:, 0], np.zeros_like(histogram_textblob_pca), color='red', label="TextBlob PCA")
    for i, lang in enumerate(languages_textblob):
        plt.text(histogram_textblob_pca[i, 0], 0.0, lang, rotation=90, va='bottom', ha='center')

    # Plot the PCA results with labels for BERT
    plt.scatter(histogram_bert_pca[:, 0], np.zeros_like(histogram_bert_pca), color='blue', label="BERT PCA")
    for i, lang in enumerate(languages_bert):
        plt.text(histogram_bert_pca[i, 0], 0.0, lang, rotation=90, va='bottom', ha='center')

    plt.xlabel('Principal Component')
    plt.legend()
    plt.show()


def umap_clustering():
    # Transform TextBlob data using UMAP
    umap_textblob = umap.UMAP(n_components=2)
    textblob = np.array(histogram_textblob).reshape(-1, 1)  # Flatten and reshape for UMAP
    histogram_textblob_umap = umap_textblob.fit_transform(textblob)

    # Transform BERT data using UMAP
    umap_bert = umap.UMAP(n_components=2)
    bert = np.array(histogram_bert).reshape(-1, 1)  # Flatten and reshape for UMAP
    histogram_bert_umap = umap_bert.fit_transform(bert)

    # Assuming you have a list of languages associated with the data
    languages_textblob = []  # List to store language labels for TextBlob
    languages_bert = []  # List to store language labels for BERT

    for i, language in enumerate(result_textblob):
        for polarity_type in ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]:
            for polarity_data in result_textblob[language][polarity_type]:
                languages_textblob.append(language)

    for i, language in enumerate(result_bert):
        for polarity_type in ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]:
            for polarity_data in result_bert[language][polarity_type]:
                languages_bert.append(language)

    # Plot the UMAP results with labels for TextBlob
    plt.figure()
    plt.scatter(histogram_textblob_umap[:, 0], histogram_textblob_umap[:, 1], color='red', label="TextBlob UMAP")
    for i, lang in enumerate(languages_textblob):
        plt.text(histogram_textblob_umap[i, 0], histogram_textblob_umap[i, 1], lang, va='bottom', ha='center',
                 rotation=90)

    # Plot the UMAP results with labels for BERT
    plt.scatter(histogram_bert_umap[:, 0], histogram_bert_umap[:, 1], color='blue', label="BERT UMAP")
    for i, lang in enumerate(languages_bert):
        plt.text(histogram_bert_umap[i, 0], histogram_bert_umap[i, 1], lang, va='bottom', ha='center',
                 rotation=90)

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.show()


def make_histogram_forAll():
    for i, language in enumerate(result_textblob):
        # Looping through different polarities and append values to langData
        for polarity_type in ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]:
            for polarity_data in result_textblob[language][polarity_type]:
                histogram_textblob.append(float(polarity_data[-5]))

    for i, language in enumerate(result_bert):
        # Looping through different polarities and append values to langData
        for polarity_type in ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]:
            for polarity_data in result_bert[language][polarity_type]:
                histogram_bert.append(float(polarity_data[-3]))

    plt.hist(histogram_textblob, bins=21, alpha=0.5, color='red', label="TextBlob")
    plt.hist(histogram_bert, bins=21, alpha=0.5, color='blue', label="BERT")

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Display the plot
    plt.show()


def histogram_separately():
    for language in result_textblob:
        textBlob = []  # polarities in TextBlob for each language
        bert = []  # polarities in BERT for each language

        # Looping through different polarities and append values to nums
        for polarity_type in ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]:
            for polarity_data in result_textblob[language][polarity_type]:
                textBlob.append(float(polarity_data[-5]))

            for polarity_data in result_bert[language][polarity_type]:
                bert.append(float(polarity_data[-3]))

        # Create a new figure for each language
        plt.figure()

        # Plot histogram for TextBlob
        plt.hist(textBlob, bins=21, alpha=0.5, color='red', label="TextBlob")
        plt.hist(bert, bins=21, alpha=0.5, color='blue', label="BERT")

        # Add labels and legend for TextBlob
        plt.title(language)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()

        # Display the plot for TextBlob
        plt.show()


def show_graph_bert():
    # set width of bar
    barWidth = 0.15
    # set height of bar
    negative = []
    weaklyNegative = []
    neutral = []
    weaklyPositive = []
    positive = []
    languageTitles = []

    for language in result_bert:
        languageTitles.append(language)
        totalNegative = len(result_bert[language]["negative"])
        totalWeaklyNegative = len(result_bert[language]["weaklyNegative"])
        totalNeutral = len(result_bert[language]["neutral"])
        totalWeaklyPositive = len(result_bert[language]["weaklyPositive"])
        totalPositive = len(result_bert[language]["positive"])
        total = totalNegative + totalWeaklyNegative + totalNeutral + totalWeaklyPositive + totalPositive
        negative.append(totalNegative / total)
        weaklyNegative.append(totalWeaklyNegative / total)
        neutral.append(totalNeutral / total)
        weaklyPositive.append(totalWeaklyPositive / total)
        positive.append(totalPositive / total)

    # Set position of bar on X axis
    br1 = np.arange(len(negative))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]

    # Make the plot
    plt.bar(br1, negative, color='red', width=barWidth,
            edgecolor='red', label='Negative')
    plt.bar(br2, weaklyNegative, color='orange', width=barWidth,
            edgecolor='orange', label='WeaklyNegative')
    plt.bar(br3, neutral, color='grey', width=barWidth,
            edgecolor='grey', label='Neutral')
    plt.bar(br4, weaklyPositive, color='blue', width=barWidth,
            edgecolor='blue', label='WeaklyPositive')
    plt.bar(br5, positive, color='green', width=barWidth,
            edgecolor='green', label='Positive')

    # Adding Xticks
    plt.xlabel('Languages', fontweight='bold', fontsize=15)
    plt.ylabel('Popularity-BERT', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(negative))], languageTitles)
    # Rotate tick labels vertically
    plt.xticks(rotation='vertical')

    plt.legend()
    plt.show()


def show_graph_textBlob():
    # set width of bar
    barWidth = 0.15
    # set height of bar
    negative = []
    weaklyNegative = []
    neutral = []
    weaklyPositive = []
    positive = []
    languageTitles = []

    for language in result_textblob:
        languageTitles.append(language)
        totalNegative = len(result_textblob[language]["negative"])
        totalWeaklyNegative = len(result_textblob[language]["weaklyNegative"])
        totalNeutral = len(result_textblob[language]["neutral"])
        totalWeaklyPositive = len(result_textblob[language]["weaklyPositive"])
        totalPositive = len(result_textblob[language]["positive"])
        total = totalNegative + totalWeaklyNegative + totalNeutral + totalWeaklyPositive + totalPositive
        negative.append(totalNegative / total)
        weaklyNegative.append(totalWeaklyNegative / total)
        neutral.append(totalNeutral / total)
        weaklyPositive.append(totalWeaklyPositive / total)
        positive.append(totalPositive / total)

    # Set position of bar on X axis
    br1 = np.arange(len(negative))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]

    # Make the plot
    plt.bar(br1, negative, color='red', width=barWidth,
            edgecolor='red', label='Negative')
    plt.bar(br2, weaklyNegative, color='orange', width=barWidth,
            edgecolor='orange', label='WeaklyNegative')
    plt.bar(br3, neutral, color='grey', width=barWidth,
            edgecolor='grey', label='Neutral')
    plt.bar(br4, weaklyPositive, color='blue', width=barWidth,
            edgecolor='blue', label='WeaklyPositive')
    plt.bar(br5, positive, color='green', width=barWidth,
            edgecolor='green', label='Positive')

    # Adding Xticks
    plt.xlabel('Languages', fontweight='bold', fontsize=15)
    plt.ylabel('Popularity-TextBlob', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(negative))], languageTitles)
    # Rotate tick labels vertically
    plt.xticks(rotation='vertical')

    plt.legend()
    plt.show()


def read_file():
    df = pd.read_excel(filename)
    # Open the xlsx file for reading
    for index, row in df.iterrows():
        textBlob = row['TextBlob Class']
        bert = row['BERT Class']
        language = row['language']

        if language not in result_textblob:
            result_textblob[language] = {"negative": [], "weaklyNegative": [], "neutral": [], "weaklyPositive": [],
                                         "positive": []}
        if language not in result_bert:
            result_bert[language] = {"negative": [], "weaklyNegative": [], "neutral": [], "weaklyPositive": [],
                                     "positive": []}

        if textBlob == 1:
            result_textblob[language]["negative"].append(row)
        elif textBlob == 2:
            result_textblob[language]["weaklyNegative"].append(row)
        elif textBlob == 3:
            result_textblob[language]["neutral"].append(row)
        elif textBlob == 4:
            result_textblob[language]["weaklyPositive"].append(row)
        elif textBlob == 5:
            result_textblob[language]["positive"].append(row)

        if bert == 1:
            result_bert[language]["negative"].append(row)
        elif bert == 2:
            result_bert[language]["weaklyNegative"].append(row)
        elif bert == 3:
            result_bert[language]["neutral"].append(row)
        elif bert == 4:
            result_bert[language]["weaklyPositive"].append(row)
        elif bert == 5:
            result_bert[language]["positive"].append(row)


def detect_classof_textBlob(polarity):
    if -1 <= polarity < -0.6:
        return 1
    elif -0.6 <= polarity < -0.2:
        return 2
    elif -0.2 <= polarity < 0.2:
        return 3
    elif 0.2 <= polarity < 0.6:
        return 4
    else:
        return 5


def detect_language(sentence):
    langs = []
    words = word_tokenize(sentence)
    for word in words:
        word = word.lower()
        if word in languages:
            if word not in langs:
                langs.append(word)
    return langs


def analyze_sentiment(sentence):
    # Creating a TextBlob object
    blob = TextBlob(sentence)

    # Getting the polarity score, which ranges from -1 (negative) to 1 (positive)
    polarity = blob.sentiment.polarity

    return polarity


def split_and_save(data):
    sentences = sent_tokenize(data['text'])
    for sentence in sentences:
        textblob_polarity = analyze_sentiment(sentence)
        textblob_class = detect_classof_textBlob(textblob_polarity)
        bert_polarity = online_sentiment_analyze(sentence)[0]['score']
        bert_class = int(online_sentiment_analyze(sentence)[0]['label'].split(' ')[0])
        languages = detect_language(sentence)
        if len(languages) > 0:
            if len(languages) > 1:
                for language in languages:
                    if "comment" in data:
                        excel_data.append(
                            {'postId': data['post'], 'commentId': data['comment'], 'sentence': sentence,
                             'TextBlob': textblob_polarity, 'TextBlob Class': textblob_class, 'BERT': bert_polarity,
                             'BERT Class': bert_class, 'language': language})
                    else:
                        excel_data.append(
                            {'postId': data['post'], 'commentId': "", 'sentence': sentence,
                             'TextBlob': textblob_polarity, 'TextBlob Class': textblob_class, 'BERT': bert_polarity,
                             'BERT Class': bert_class, 'language': language})
            else:
                if "comment" in data:
                    excel_data.append(
                        {'postId': data['post'], 'commentId': data['comment'], 'sentence': sentence,
                         'TextBlob': textblob_polarity, 'TextBlob Class': textblob_class, 'BERT': bert_polarity,
                         'BERT Class': bert_class, 'language': languages[0]})
                else:
                    excel_data.append(
                        {'postId': data['post'], 'commentId': "", 'sentence': sentence,
                         'TextBlob': textblob_polarity, 'TextBlob Class': textblob_class, 'BERT': bert_polarity,
                         'BERT Class': bert_class, 'language': languages[0]})
    analiz_df = pd.DataFrame(excel_data)
    analiz_df.to_excel(filename, index=False)


def process_data():
    for post in posts:
        if post.selftext != '':
            split_and_save({"post": post, "text": post.selftext.replace('\n', ' ')})
        if len(post.comments.list()) != 0:
            for comment in post.comments.list():
                if isinstance(comment, praw.models.Comment):
                    split_and_save({"post": post, "comment": comment, "text": comment.body.replace('\n', ' ')})


def pull_data_from_api():
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

    # looping over posts and scraping it
    for post in subreddit.new(limit=None):
        posts.append(post)
        # Introduce a delay between requests (e.g., 5 seconds)
        time.sleep(10)


pull_data_from_api()
process_data()
# read_file()
# show_graph_textBlob()
# show_graph_bert()
# histogram_separately()
# make_histogram_forAll()
# umap_clustering()
# pca()
