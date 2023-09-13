import praw
from textblob import TextBlob
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# results
# {"C":{negative:[], weaklyNegative:[], neutral:[], weaklyPositive:[], positive:[]}, "Python":{negative:[], weaklyNegative:[], neutral:[], weaklyPositive:[], positive:[]},}
result = {}

colorsArray = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "black",
    "cyan",
    "magenta",
    "teal",
    "lavender",
    "lime",
    "olive",
    "maroon",
    "navy",
    "turquoise",
    "indigo",
    "violet",
    "beige",
    "coral",
    "gold",
    "silver",
    "peru",
    "orchid",
    "plum",
    "aqua",
    "chartreuse",
    "crimson",
    "fuchsia",
    "khaki",
    "limegreen",
    "navajowhite",
    "orangered",
    "royalblue",
    "saddlebrown",
    "salmon",
    "seagreen",
    "sienna",
    "slateblue",
    "springgreen",
    "tan",
    "thistle",
    "tomato",
    "wheat",
    "yellowgreen",
    "darkcyan",
    "deepskyblue"
]


def pca2():
    # Initialize empty lists or arrays for each category and language
    categories = ["negative", "weaklyNegative", "neutral", "weaklyPositive", "positive"]
    languages = list(result.keys())
    polarity_data = {category: {language: [] for language in languages} for category in categories}

    # Extract polarities for each sentence and categorize by language and category
    for language, categories_data in result.items():
        for category, sentences in categories_data.items():
            for sentence_obj in sentences:
                polarity_data[category][language].append(sentence_obj[3])  # Assuming polarity is at index 3

    # Standardize data
    standardized_data = {}
    for category, language_polarities in polarity_data.items():
        for language, polarities in language_polarities.items():
            standardized_data[f"{language}_{category}"] = StandardScaler().fit_transform(
                np.array(polarities).reshape(-1, 1)
            )

    # Perform PCA analysis
    pca_results = {}
    for category_language, data_array in standardized_data.items():
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data_array)
        pca_results[category_language] = reduced_data

    # Visualize the PCA results
    for category_language, reduced_data in pca_results.items():
        category, language = category_language.split("_")
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label=f"{language} - {category}")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.title("PCA Analysis of Polarities")
    plt.grid()
    plt.show()


# pca with total number of negative, weaklyNegative, neutral, weaklyPositive, positive
def pca():
    data = {"language": [], "negativeNum": [], "weaklyNegativeNum": [], "neutral": [], "weaklyPositive": [],
            "positive": []}

    for res in result:
        lang = res
        negNumber = len(result[res]["negative"])
        weaklyNegNumber = len(result[res]["weaklyNegative"])
        neutral = len(result[res]["neutral"])
        weaklyPosNumber = len(result[res]["weaklyPositive"])
        posNumber = len(result[res]["positive"])
        data["language"].append(lang)
        data["negativeNum"].append(negNumber)
        data["weaklyNegativeNum"].append(weaklyNegNumber)
        data["neutral"].append(neutral)
        data["weaklyPositive"].append(weaklyPosNumber)
        data["positive"].append(posNumber)

    # print(lang, " ", negNumber, " ", weaklyNegNumber, " ", neutral, " ", weaklyPosNumber, " ", posNumber)

    df = pd.DataFrame(data)

    # Dil sütununu indeks olarak ayarla
    df.set_index('language', inplace=True)

    # Verileri standartlaştır
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # PCA uygula
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    # İki bileşeni görselleştir
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Dil etiketlerini ekle
    for i, lang in enumerate(df.index):
        plt.annotate(lang, (reduced_data[i, 0], reduced_data[i, 1]))

    plt.title('PCA Analysis Result')
    plt.grid()
    plt.show()


def make_histogram_forAll():
    for i, language in enumerate(result):
        langData = []
        for negative in result[language]["negative"]:
            langData.append(float(negative[-2]))
        for weaklyNeg in result[language]["weaklyNegative"]:
            langData.append(float(weaklyNeg[-2]))
        for neutral in result[language]["neutral"]:
            langData.append(float(neutral[-2]))
        for weaklyPos in result[language]["weaklyPositive"]:
            langData.append(float(weaklyPos[-2]))
        for positive in result[language]["positive"]:
            langData.append(float(positive[-2]))

        plt.hist(langData, bins=5, alpha=0.5, color=colorsArray[i], label=language)

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Display the plot
    plt.show()


def make_histogram_seperately():
    nums = []
    for language in result:
        for negative in result[language]["negative"]:
            nums.append(float(negative[-2]))
        for weaklyNeg in result[language]["weaklyNegative"]:
            nums.append(float(weaklyNeg[-2]))
        for neutral in result[language]["neutral"]:
            nums.append(float(neutral[-2]))
        for weaklyPos in result[language]["weaklyPositive"]:
            nums.append(float(weaklyPos[-2]))
        for positive in result[language]["positive"]:
            nums.append(float(positive[-2]))

        # Creating dataset
        n_bins = 5

        legend = ['distribution']

        # Creating histogram
        fig, axs = plt.subplots(1, 1,
                                figsize=(10, 5),
                                tight_layout=True)

        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            axs.spines[s].set_visible(False)

        # Remove x, y ticks
        axs.xaxis.set_ticks_position('none')
        axs.yaxis.set_ticks_position('none')

        # Add padding between axes and labels
        axs.xaxis.set_tick_params(pad=5)
        axs.yaxis.set_tick_params(pad=10)

        # Creating histogram
        N, bins, patches = axs.hist(nums, bins=n_bins, range=(-1, 1))  # Set the range to -1 to 1

        # Setting color
        fracs = ((N ** (1 / 5)) / N.max())
        norm = colors.Normalize(fracs.min(), fracs.max())

        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        # Adding extra features
        plt.xlabel("Polarities")
        plt.ylabel("Number of sentences")
        plt.legend(legend)
        plt.title(language)

        # Show plot
        plt.show()


def show_graph_without_dividing():
    # set width of bar
    barWidth = 0.15
    fig = plt.subplots(figsize=(12, 8))
    # set height of bar
    negative = []
    weaklyNegative = []
    neutral = []
    weaklyPositive = []
    positive = []
    languageTitles = []

    for language in result:
        languageTitles.append(language)
        totalNegative = len(result[language]["negative"])
        totalWeaklyNegative = len(result[language]["weaklyNegative"])
        totalNeutral = len(result[language]["neutral"])
        totalWeaklyPositive = len(result[language]["weaklyPositive"])
        totalPositive = len(result[language]["positive"])
        negative.append(totalNegative)
        weaklyNegative.append(totalWeaklyNegative)
        neutral.append(totalNeutral)
        weaklyPositive.append(totalWeaklyPositive)
        positive.append(totalPositive)

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
    plt.ylabel('Popularity', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(negative))], languageTitles)

    # Rotate tick labels vertically
    plt.xticks(rotation='vertical')

    plt.legend()
    plt.show()


def show_graph():
    # set width of bar
    barWidth = 0.15
    fig = plt.subplots(figsize=(12, 8))
    # set height of bar
    negative = []
    weaklyNegative = []
    neutral = []
    weaklyPositive = []
    positive = []
    languageTitles = []

    for language in result:
        languageTitles.append(language)
        totalNegative = len(result[language]["negative"])
        totalWeaklyNegative = len(result[language]["weaklyNegative"])
        totalNeutral = len(result[language]["neutral"])
        totalWeaklyPositive = len(result[language]["weaklyPositive"])
        totalPositive = len(result[language]["positive"])
        total = totalNegative + totalWeaklyNegative + totalNeutral + totalWeaklyPositive + totalPositive
        print(language, "  :  ", total)
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
    plt.ylabel('Popularity', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(negative))], languageTitles)
    # Rotate tick labels vertically
    plt.xticks(rotation='vertical')

    plt.legend()
    plt.show()


def read_file():
    # Open the CSV file for reading
    with open(filename, mode='r', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)

        # Loop through each row in the CSV file
        for row in reader:
            language = row[-1]
            polarity = float(row[-2])
            if language not in result:
                result[language] = {"negative": [], "weaklyNegative": [], "neutral": [], "weaklyPositive": [],
                                    "positive": []}

            if -1 <= polarity <= -0.6:
                result[language]["negative"].append(row)
            elif -0.6 <= polarity <= -0.2:
                result[language]["weaklyNegative"].append(row)
            elif -0.2 <= polarity <= 0.2:
                result[language]["neutral"].append(row)
            elif 0.2 <= polarity <= 0.6:
                result[language]["weaklyPositive"].append(row)
            else:
                result[language]["positive"].append(row)


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


filename = "allDataCsv.csv"


def split_and_save(data):
    # writing to csv file
    with open(filename, 'a', encoding='utf-8', newline='') as csvfile:
        # creating a csv writer object
        file = csv.writer(csvfile)

        sentences = sent_tokenize(data['text'])
        for sentence in sentences:
            sentiment = analyze_sentiment(sentence)
            languages = detect_language(sentence)
            if len(languages) > 0:
                if len(languages) > 1:
                    for language in languages:
                        if "comment" in data:
                            file.writerow([data['post'], data['comment'], sentence, sentiment, language])
                        else:
                            file.writerow([data['post'], "", sentence, sentiment, language])
                else:
                    if "comment" in data:
                        file.writerow([data['post'], data['comment'], sentence, sentiment, languages[0]])
                    else:
                        file.writerow([data['post'], "", sentence, sentiment, languages[0]])


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


# pull_data_from_api()
# process_data()
read_file()
# show_graph()
# show_graph_without_dividing()
# make_histogram_seperately()
# make_histogram_forAll()
# pca()
pca2()