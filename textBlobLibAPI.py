import praw
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
from nltk import word_tokenize
from tabulate import tabulate

# posts
posts = []

# languages
languages = [
    "ABAP",
    "ActionScript",
    "Ada",
    "ALGOL",
    "Alice",
    "APL",
    "Assembly",
    "AutoIt",
    "AutoLISP",
    "Bash",
    "C",
    "C#",
    "C++",
    "COBOL",
    "Clojure",
    "COOL",
    "Crystal",
    "D",
    "Dart",
    "Delphi",
    "Eiffel",
    "Elixir",
    "Elm",
    "Erlang",
    "F#",
    "Forth",
    "Fortran",
    "Go",
    "Groovy",
    "Haskell",
    "HTML",
    "Java",
    "JavaScript",
    "Julia",
    "Kotlin",
    "Lisp",
    "Lua",
    "MATLAB",
    "Objective-C",
    "Pascal",
    "Perl",
    "PHP",
    "Prolog",
    "Python",
    "R",
    "Ruby",
    "Rust",
    "Scala",
    "Scheme",
    "Shell",
    "Swift",
    "Tcl",
    "TypeScript",
    "VBScript",
    "Verilog",
    "VHDL",
    "Visual Basic .NET"
]

# languages for Posts {python:{post1, post3}, javascript:{post2}, java:{post4, post5}}
languagePosts = {}

# languages for comments of post {python:{post1comment1, post3comment2}, javascript:{post2comment1}, java:{post4comment5, post5comment6}}
languageComments = {}

# {language1:{positiveSentences:[obj, obj], negativeSentences:[obj, obj], neutralSentences:[obj, obj]}, language2:{}}
# obj: {sentence: '', polarity: ''}
results = {}

# for printing data in a table
data = [['SENTENCE', "LANGUAGE", "SENTIMENT"], ['________', "________", "_________"], ]

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
for post in subreddit.new(limit=5):
    posts.append(post)


# make language-post analysis
def make_post_content_analysis():
    for post in posts:
        content = post.selftext
        words = word_tokenize(content)
        for word in words:
            if word in languages:
                if word not in languagePosts:
                    languagePosts[word] = []
                comments = []
                post.comments.replace_more(limit=None)  # Fetch all comments
                for comment in post.comments.list():
                    if isinstance(comment, praw.models.Comment):
                        comments.append(comment.body)

                languagePosts[word].append(
                    {"title": post.title, "content": post.selftext, "comments": comments})


# make language-post-comment analysis
def make_post_comment_analysis():
    for key, value in languagePosts.items():
        for post in value:
            comment_list = post["comments"]
            for comment in comment_list:
                words = word_tokenize(comment)
                for word in words:
                    if word in languages:
                        if word not in languageComments:
                            languageComments[word] = []
                        languageComments[word].append(comment)


# analyzing the sentence if it is positive, negative or neutral
def analyze_sentiment(sentence):
    # Creating a TextBlob object
    blob = TextBlob(sentence)

    # Getting the polarity score, which ranges from -1 (negative) to 1 (positive)
    polarity = blob.sentiment.polarity

    return polarity


def make_analysis():
    for key, value in languageComments.items():
        for comment in value:
            words = word_tokenize(comment)
            for word in words:
                if word in languages:
                    if word not in results:
                        results[word] = {"positiveSentences": [], "weaklyPositiveSentences": [],
                                         "negativeSentences": [], "weaklyNegativeSentences": [], "neutralSentences": []}

                    polarity = analyze_sentiment(comment.lower())
                    sentenceObj = {"sentence": comment, "polarity": polarity}

                    if -1 <= polarity <= -0.6:
                        results[word]["negativeSentences"].append(sentenceObj)
                    elif -0.6 <= polarity <= -0.2:
                        results[word]["weaklyNegativeSentences"].append(sentenceObj)
                    elif -0.2 <= polarity <= 0.2:
                        results[word]["neutralSentences"].append(sentenceObj)
                    elif 0.2 <= polarity <= 0.6:
                        results[word]["weaklyPositiveSentences"].append(sentenceObj)
                    else:
                        results[word]["positiveSentences"].append(sentenceObj)

                    row = [comment, word, polarity]
                    data.append(row)


def showGraph():
    # x-coordinates of left sides of bars
    left = np.arange(len(results))

    # heights of bars
    heightPositive = []
    heightWeaklyPositive = []
    heightNegative = []
    heightWeaklyNegative = []
    heightNeutral = []

    # labels for bars
    tick_label = []

    # widths of the bars
    bar_width = 0.5

    for data in results:
        heightNegative.append(len(results[data]["negativeSentences"]))
        heightWeaklyNegative.append(len(results[data]["weaklyNegativeSentences"]))
        heightNeutral.append(len(results[data]["neutralSentences"]))
        heightWeaklyPositive.append(len(results[data]["weaklyPositiveSentences"]))
        heightPositive.append(len(results[data]["positiveSentences"]))
        tick_label.append(data)

    total = heightNegative + heightWeaklyNegative + heightNeutral + heightWeaklyPositive + heightPositive
    # Plotting the bars for positive and negative sentiments side by side
    plt.bar(left - 2 * bar_width, heightNegative, width=bar_width, label='Negative', color='red')
    plt.bar(left - bar_width, heightWeaklyNegative, width=bar_width, label='WeaklyNegative', color='orange')
    plt.bar(left, heightNeutral, width=bar_width, label='Neutral', color='gray')
    plt.bar(left + bar_width, heightWeaklyPositive, width=bar_width, label='WeaklyPositive', color='green')
    plt.bar(left + 2 * bar_width, heightPositive, width=bar_width, label='Positive', color='blue')

    # naming the x-axis
    plt.xlabel('Languages')
    # naming the y-axis
    plt.ylabel('Number of Sentences')
    # plot title
    plt.title('Sentiments for Each Language')

    # setting the x-ticks to be at the middle of each group of bars
    plt.xticks(left + bar_width, tick_label)

    # Rotate tick labels vertically
    plt.xticks(rotation='vertical')

    # displaying the legend
    plt.legend()

    # function to show the plot
    plt.show()


make_post_content_analysis()
make_post_comment_analysis()
make_analysis()
# print(tabulate(data))
showGraph()
# print(languagePosts)
# print(languageComments)


# post başlığı: post.title
# post texti: post.selftext
# post yorumları: post.comment.list()
# yorum texti: comment.body

# her postun yorumları
# for comment in submission.comments.list():
# print(comment.body)
