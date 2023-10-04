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
import seaborn as sns

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

pulled_data_file = "pulledData.xlsx"
processed_data_file = "processedData.xlsx"
textblob_languages_file = "textblobLanguages.xlsx"
bert_languages_file = "bertLanguages.xlsx"
textblob_sentiments_for_each_lang_file = "textblobSentimentsForEachLang.xlsx"
bert_sentiments_for_each_lang_file = "bertSentimentsForEachLang.xlsx"

# posts
posts = []
pulled_data = []
processed_data = []

bert_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


def show_umap():
    textblob_data = store_each_lang_sentiments()["textblob"]
    bert_data = store_each_lang_sentiments()["bert"]

    # Find the maximum length of arrays
    max_length = max(max(len(arr) for arr in textblob_data.values()),
                     max(len(arr) for arr in bert_data.values()))

    # Pad arrays to the maximum length with zeros
    for lang in textblob_data:
        textblob_data[lang] += [0.0] * (max_length - len(textblob_data[lang]))

    for cluster in bert_data:
        bert_data[cluster] += [0.0] * (max_length - len(bert_data[cluster]))

    # Convert the dictionary values to a list of arrays
    textblob_arrays = np.array([textblob_data[lang] for lang in textblob_data])
    bert_arrays = np.array([bert_data[cluster] for cluster in bert_data])

    # Apply UMAP for both textblob and BERT data with a fixed random_state
    umap_textblob = umap.UMAP(n_components=2, random_state=42)
    umap_result_textblob = umap_textblob.fit_transform(textblob_arrays)

    umap_bert = umap.UMAP(n_components=2, random_state=42)
    umap_result_bert = umap_bert.fit_transform(bert_arrays)

    # Get the UMAP components for both textblob and BERT data
    umap1_textblob = umap_result_textblob[:, 0]
    umap2_textblob = umap_result_textblob[:, 1]

    umap1_bert = umap_result_bert[:, 0]
    umap2_bert = umap_result_bert[:, 1]

    # Get the language and cluster names
    languages = list(textblob_data.keys())
    clusters = list(bert_data.keys())

    # Plot the UMAP embeddings for textblob data
    plt.figure(figsize=(10, 8))
    plt.scatter(umap1_textblob, umap2_textblob, label='TextBlob Data')

    # Annotate each point with the language name
    for i, lang in enumerate(languages):
        plt.annotate(lang, (umap1_textblob[i], umap2_textblob[i]))

    # Plot the UMAP embeddings for BERT data
    plt.scatter(umap1_bert, umap2_bert, label='BERT Data')

    # Annotate each point with the cluster name
    for i, cluster in enumerate(clusters):
        plt.annotate(cluster, (umap1_bert[i], umap2_bert[i]))

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Visualization of TextBlob and BERT Data')
    plt.legend()
    plt.show()


def color_palet():
    textblob_data = store_each_lang_sentiments()["textblob"]
    bert_data = store_each_lang_sentiments()["bert"]

    # Find the maximum length of arrays
    max_length = max(max(len(arr) for arr in textblob_data.values()),
                     max(len(arr) for arr in bert_data.values()))

    # Pad arrays to the maximum length with zeros
    for lang in textblob_data:
        textblob_data[lang] += [0.0] * (max_length - len(textblob_data[lang]))

    for cluster in bert_data:
        bert_data[cluster] += [0.0] * (max_length - len(bert_data[cluster]))

    # Convert the dictionary values to a list of arrays
    textblob_arrays = np.array([textblob_data[lang] for lang in textblob_data])
    bert_arrays = np.array([bert_data[cluster] for cluster in bert_data])

    # Apply UMAP for both textblob and BERT data with a fixed random_state
    umap_textblob = umap.UMAP(n_components=2, random_state=42)
    umap_result_textblob = umap_textblob.fit_transform(textblob_arrays)

    umap_bert = umap.UMAP(n_components=2, random_state=42)
    umap_result_bert = umap_bert.fit_transform(bert_arrays)

    # Get the language and cluster names
    languages = list(textblob_data.keys())
    clusters = list(bert_data.keys())

    # Define a color palette for clusters
    cluster_palette = sns.color_palette("husl", len(clusters))

    # Plot the UMAP embeddings for textblob data
    plt.figure(figsize=(10, 8))

    for i, cluster in enumerate(clusters):
        cluster_points_bert = bert_data[cluster]
        num_points = len(cluster_points_bert)
        brightness = np.linspace(0.2, 1, num_points)

        # Assign a color to each point within the cluster
        cluster_colors = [sns.set_hls_values(cluster_palette[i], l=b) for b in brightness]

        # Plot the points for each cluster with varying brightness
        plt.scatter(umap_result_bert[:num_points, 0], umap_result_bert[:num_points, 1],
                    color=cluster_colors,
                    label=f'Cluster {cluster}')

    # Annotate each point with the language name
    for i, lang in enumerate(languages):
        plt.annotate(lang, (umap_result_textblob[i, 0], umap_result_textblob[i, 1]))

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Visualization of TextBlob and BERT Data')
    plt.legend()
    plt.show()


def pca_bert_specific_lang():
    textblob_data = store_each_lang_sentiments()["textblob"]
    bert_data = store_each_lang_sentiments()["bert"]

    # Find the maximum length of arrays
    max_length = max(max(len(arr) for arr in textblob_data.values()),
                     max(len(arr) for arr in bert_data.values()))

    # Pad arrays to the maximum length with zeros
    for lang in textblob_data:
        textblob_data[lang] += [0.0] * (max_length - len(textblob_data[lang]))

    for cluster in bert_data:
        bert_data[cluster] += [0.0] * (max_length - len(bert_data[cluster]))

    # Convert the dictionary values to a list of arrays
    bert_arrays = np.array([bert_data[cluster] for cluster in bert_data])

    # Apply PCA for BERT data
    pca_bert = PCA(n_components=2)
    pca_result_bert = pca_bert.fit_transform(bert_arrays)

    pc1_bert = pca_result_bert[:, 0]

    # Get the language and cluster names
    languages = list(textblob_data.keys())

    # Calculate average PCA1_BERT for each language
    avg_pca1_bert_lang = np.mean(pca_result_bert, axis=1)

    # Plot the average BERT-specific language and PCA1_BERT values
    plt.figure(figsize=(10, 8))
    plt.scatter(pc1_bert, avg_pca1_bert_lang, color='red', label='Average BERT Data')

    # Annotate each language point with the language name
    for i, lang in enumerate(languages):
        plt.annotate(lang, (pc1_bert[i], avg_pca1_bert_lang[i]))

    plt.xlabel('PCA1_BERT')
    plt.ylabel('Average PCA1_BERT for Each Language')
    plt.title('Average PCA1_BERT for Each Language vs. PCA1_BERT')
    plt.legend()
    plt.show()


def pca():
    textblob_data = store_each_lang_sentiments()["textblob"]
    bert_data = store_each_lang_sentiments()["bert"]

    # Find the maximum length of arrays
    max_length = max(max(len(arr) for arr in textblob_data.values()),
                     max(len(arr) for arr in bert_data.values()))

    # Pad arrays to the maximum length with zeros
    for lang in textblob_data:
        textblob_data[lang] += [0.0] * (max_length - len(textblob_data[lang]))

    for cluster in bert_data:
        bert_data[cluster] += [0.0] * (max_length - len(bert_data[cluster]))

    # Convert the dictionary values to a list of arrays
    textblob_arrays = np.array([textblob_data[lang] for lang in textblob_data])
    bert_arrays = np.array([bert_data[cluster] for cluster in bert_data])

    # Apply PCA for both textblob and BERT data
    pca_textblob = PCA(n_components=2)
    pca_result_textblob = pca_textblob.fit_transform(textblob_arrays)

    pca_bert = PCA(n_components=2)
    pca_result_bert = pca_bert.fit_transform(bert_arrays)

    # Get the principal components for both textblob and BERT data
    pc1_textblob = pca_result_textblob[:, 0]
    pc2_textblob = pca_result_textblob[:, 1]

    pc1_bert = pca_result_bert[:, 0]
    pc2_bert = pca_result_bert[:, 1]

    # Get the language and cluster names
    languages = list(textblob_data.keys())
    clusters = list(bert_data.keys())

    # Plot the languages based on the first two principal components
    plt.figure(figsize=(10, 8))
    plt.scatter(pc1_textblob, pc2_textblob, label='TextBlob Data')

    # Annotate each language point with the language name
    for i, lang in enumerate(languages):
        plt.annotate(lang, (pc1_textblob[i], pc2_textblob[i]))

    # Plot the clusters based on the first two principal components
    plt.scatter(pc1_bert, pc2_bert, label='BERT Data')

    # Annotate each cluster point with the cluster name
    for i, cluster in enumerate(clusters):
        plt.annotate(cluster, (pc1_bert[i], pc2_bert[i]))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Analysis of TextBlob and BERT Data')
    plt.legend()
    plt.show()


def store_each_lang_sentiments():
    res_textblob = {}
    res_bert = {}

    df = pd.read_excel(processed_data_file)
    for index, row in df.iterrows():
        language = row['language']
        textblob = row["TextBlob"]
        bert = row["BERT"]

        if language not in res_textblob:
            res_textblob[language] = []
            res_bert[language] = []

        res_textblob[language].append(textblob)
        res_bert[language].append(bert)

    # analiz_df = pd.DataFrame(res_textblob)
    # analiz_df.to_excel(textblob_sentiments_for_each_lang_file, index=False)

    # store_bert = pd.DataFrame(res_bert)
    # store_bert.to_excel(bert_sentiments_for_each_lang, index=False)

    return {"textblob": res_textblob, "bert": res_bert}


def histogram_separately():
    res_textblob = store_each_lang_sentiments()["textblob"]
    res_bert = store_each_lang_sentiments()["bert"]
    for language in res_textblob:
        plt.figure()

        # Plot histograms
        plt.hist(res_textblob[language], bins=21, alpha=0.5, color='red', label="TextBlob")
        plt.hist(res_bert[language], bins=21, alpha=0.5, color='blue', label="BERT")

        plt.title(language)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()

        plt.show()


def make_histogram_forAll():
    histogram_textblob = []  # for polarity values for each language
    histogram_bert = []  # for polarity values for each language

    df = pd.read_excel(processed_data_file)
    for index, row in df.iterrows():
        textblob = row['TextBlob']
        bert = row["BERT"]
        histogram_textblob.append(float(textblob))
        histogram_bert.append(float(bert))

    plt.hist(histogram_textblob, bins=21, alpha=0.5, color='red', label="TextBlob")
    plt.hist(histogram_bert, bins=21, alpha=0.5, color='blue', label="BERT")

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Display the plot
    plt.show()


def show_bert_graph_and_save():
    result_bert = {}

    df = pd.read_excel(processed_data_file)
    for index, row in df.iterrows():
        language = row['language']
        bert_class = row['BERT Class']

        if language not in result_bert:
            result_bert[language] = {"negative": 0, "weaklyNegative": 0, "neutral": 0,
                                     "weaklyPositive": 0, "positive": 0, "total": 0}

        if bert_class == 1:
            result_bert[language]["negative"] += 1
        elif bert_class == 2:
            result_bert[language]["weaklyNegative"] += 1
        elif bert_class == 3:
            result_bert[language]["neutral"] += 1
        elif bert_class == 4:
            result_bert[language]["weaklyPositive"] += 1
        elif bert_class == 5:
            result_bert[language]["positive"] += 1

    languageTitles = []
    # set height of bar
    negative = []
    weaklyNegative = []
    neutral = []
    weaklyPositive = []
    positive = []

    for language in result_bert:
        totalNegative = result_bert[language]["negative"]
        totalWeaklyNegative = result_bert[language]["weaklyNegative"]
        totalNeutral = result_bert[language]["neutral"]
        totalWeaklyPositive = result_bert[language]["weaklyPositive"]
        totalPositive = result_bert[language]["positive"]
        total = totalNegative + totalWeaklyNegative + totalNeutral + totalWeaklyPositive + totalPositive
        result_bert[language]["total"] = total

        languageTitles.append(language)
        negative.append(totalNegative / total)
        weaklyNegative.append(totalWeaklyNegative / total)
        neutral.append(totalNeutral / total)
        weaklyPositive.append(totalWeaklyPositive / total)
        positive.append(totalPositive / total)

    # Create a DataFrame from the flattened result_bert dictionary
    flat_data = []
    for language, counts in result_bert.items():
        flat_data.append({'Language': language, **counts})

    analiz_df = pd.DataFrame(flat_data)
    analiz_df.to_excel(bert_languages_file, index=False)

    # set width of bar
    barWidth = 0.15

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


def show_textblob_graph_and_save():
    result_textblob = {}

    df = pd.read_excel(processed_data_file)
    for index, row in df.iterrows():
        language = row['language']
        textblob_class = row['TextBlob Class']

        if language not in result_textblob:
            result_textblob[language] = {"negative": 0, "weaklyNegative": 0, "neutral": 0,
                                         "weaklyPositive": 0, "positive": 0, "total": 0}

        if textblob_class == 1:
            result_textblob[language]["negative"] += 1
        elif textblob_class == 2:
            result_textblob[language]["weaklyNegative"] += 1
        elif textblob_class == 3:
            result_textblob[language]["neutral"] += 1
        elif textblob_class == 4:
            result_textblob[language]["weaklyPositive"] += 1
        elif textblob_class == 5:
            result_textblob[language]["positive"] += 1

    languageTitles = []
    # set height of bar
    negative = []
    weaklyNegative = []
    neutral = []
    weaklyPositive = []
    positive = []

    for language in result_textblob:
        totalNegative = result_textblob[language]["negative"]
        totalWeaklyNegative = result_textblob[language]["weaklyNegative"]
        totalNeutral = result_textblob[language]["neutral"]
        totalWeaklyPositive = result_textblob[language]["weaklyPositive"]
        totalPositive = result_textblob[language]["positive"]
        total = totalNegative + totalWeaklyNegative + totalNeutral + totalWeaklyPositive + totalPositive
        result_textblob[language]["total"] = total

        languageTitles.append(language)
        negative.append(totalNegative / total)
        weaklyNegative.append(totalWeaklyNegative / total)
        neutral.append(totalNeutral / total)
        weaklyPositive.append(totalWeaklyPositive / total)
        positive.append(totalPositive / total)

    # Create a DataFrame from the flattened result_textblob dictionary
    flat_data = []
    for language, counts in result_textblob.items():
        flat_data.append({'Language': language, **counts})

    analiz_df = pd.DataFrame(flat_data)
    analiz_df.to_excel(textblob_languages_file, index=False)

    # set width of bar
    barWidth = 0.15

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


def show_graphs():
    show_textblob_graph_and_save()
    show_bert_graph_and_save()


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


def find_bert_sentiment_analyzes(sentence):
    max_token_limit = 512

    # Split the sentence into chunks of 512 tokens or less
    chunks = [sentence[i:i + max_token_limit] for i in range(0, len(sentence), max_token_limit)]

    # Initialize lists to store sentiment scores and classes for each chunk
    chunk_bert_scores = []
    chunk_bert_classes = []

    # Process each chunk separately using the BERT model
    for chunk in chunks:
        result = bert_analysis(chunk)
        chunk_bert_scores.extend([res['score'] for res in result])
        chunk_bert_classes.extend([int(res['label'].split(' ')[0]) for res in result])

    # print(chunk_bert_scores)
    # print(chunk_bert_classes)
    # Calculate the average sentiment scores and classes for each chunk
    average_chunk_scores = [
        sum(chunk_bert_scores[i:i + max_token_limit]) / len(chunk_bert_scores[i:i + max_token_limit])
        for i in range(0, len(chunk_bert_scores), max_token_limit)]

    # Calculate average chunk classes
    average_chunk_classes = [
        sum(chunk_bert_classes[i:i + max_token_limit]) / len(chunk_bert_classes[i:i + max_token_limit])
        for i in range(0, len(chunk_bert_classes), max_token_limit)]

    # print("Average chunk scores:", average_chunk_scores[0])
    # print("Average chunk classes:", int(average_chunk_classes[0]))
    return ({"score": average_chunk_scores[0], "class": int(average_chunk_classes[0])})


def find_textblob_class(polarity):
    if -1 <= polarity < -0.6:
        return 1
    elif -0.6 <= polarity < -0.2:
        return 2
    elif -0.2 < polarity < 0.2:
        return 3
    elif 0.2 < polarity < 0.6:
        return 4
    else:
        return 5


def find_textblob_continuous(sentence):
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    return polarity


def make_sentiment_analyzes_and_save():
    df = pd.read_excel(pulled_data_file)
    # Open the xlsx file for reading
    for index, row in df.iterrows():
        postId = row["postId"]
        commentId = row["commentId"]
        sentence = row['sentence']
        language = row['language']
        textblob_continuous = find_textblob_continuous(sentence)
        textblob_class = find_textblob_class(textblob_continuous)
        bert_res = find_bert_sentiment_analyzes(sentence)
        bert_continuous = bert_res["score"]
        bert_class = bert_res["class"]
        # find_bert_sentiment_analyzes(sentence)[0]['score']
        # bert_class = int(find_bert_sentiment_analyzes(sentence)[0]['label'].split(' ')[0])

        processed_data.append({'postId': postId, 'commentId': commentId, 'sentence': sentence, 'language': language,
                               'TextBlob': textblob_continuous, 'TextBlob Class': textblob_class,
                               'BERT': bert_continuous,
                               'BERT Class': bert_class})

    analiz_df = pd.DataFrame(processed_data)
    analiz_df.to_excel(processed_data_file, index=False)


def detect_language(sentence):
    langs = []
    words = word_tokenize(sentence)
    for word in words:
        word = word.lower()
        if word in languages:
            if word not in langs:
                langs.append(word)
    return langs


def break_into_sentences(data):
    sentences = sent_tokenize(data['text'])
    for sentence in sentences:
        languages = detect_language(sentence)
        if len(languages) > 0:
            if len(languages) > 1:
                for language in languages:
                    if "comment" in data:
                        pulled_data.append(
                            {'postId': data['post'], 'commentId': data['comment'], 'sentence': sentence,
                             'language': language})
                    else:
                        pulled_data.append(
                            {'postId': data['post'], 'commentId': "", 'sentence': sentence, 'language': language})
            else:
                if "comment" in data:
                    pulled_data.append(
                        {'postId': data['post'], 'commentId': data['comment'], 'sentence': sentence,
                         'language': languages[0]})
                else:
                    pulled_data.append(
                        {'postId': data['post'], 'commentId': "", 'sentence': sentence, 'language': languages[0]})
    analiz_df = pd.DataFrame(pulled_data)
    analiz_df.to_excel(pulled_data_file, index=False)


def pull_data_from_api_and_save():
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
        if post.selftext != '':
            break_into_sentences({"post": post, "text": post.selftext.replace('\n', ' ')})
        if len(post.comments.list()) != 0:
            for comment in post.comments.list():
                if isinstance(comment, praw.models.Comment):
                    break_into_sentences({"post": post, "comment": comment, "text": comment.body.replace('\n', ' ')})
        # Introduce a delay between requests
        time.sleep(10)


# pull_data_from_api_and_save()
# make_sentiment_analyzes_and_save()
# show_graphs()
# make_histogram_forAll()
# histogram_separately()
# pca()
# pca_bert_specific_lang()
color_palet()
# show_umap()
