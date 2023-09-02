import praw
from textblob import TextBlob
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import csv
import time

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

# posts
posts = []

# results
# {"C":{negative:[], weaklyNegative:[], neutral:[], weaklyPositive:[], positive:[]}, "Python":{negative:[], weaklyNegative:[], neutral:[], weaklyPositive:[], positive:[]},}
result = {}

def showGraph():


def read_file_and_show_result():
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

    showGraph()


def detect_language(sentence):
    langs = []
    words = word_tokenize(sentence)
    for word in words:
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


filename = "csvData.csv"


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
    for post in subreddit.new(limit=10):
        posts.append(post)
        # Introduce a delay between requests (e.g., 5 seconds)
        time.sleep(5)


# pull_data_from_api()
# process_data()
read_file_and_show_result()
print(result)