from textblob import TextBlob
import praw


class Language:
    # negative is between -1 and -0.6
    # weaklyNegative is between -0.6 and -0.2
    # neutral is between -0.2 and 0.2
    # weaklyPositive is between 0.2 and 0.6
    # positive is between 0.6 and 1
    def __init__(self, name):
        self.name = name
        self.negative = []
        self.weaklyNegative = []
        self.neutral = []
        self.weaklyPositive = []
        self.positive = []


class Post:
    def __init__(self, post, comments, polarity=0, language=""):
        self.post = post
        self.comments = comments
        self.polarity = polarity
        self.language = language


class Comment:
    def __init__(self, comment, polarity=0, language=""):
        self.comment = comment
        self.polarity = polarity
        self.language = language


languages = []
posts = []
comments = []


# sentiment analyzer
def analyze_sentiment(sentence):
    # Creating a TextBlob object
    blob = TextBlob(sentence)

    # Getting the polarity score, which ranges from -1 (negative) to 1 (positive)
    polarity = blob.sentiment.polarity

    return polarity


# Create an instance of reddit class
reddit = praw.Reddit(username="eesraakgull",
                     password="Esra.1998",
                     client_id="dBScuPJw95o8DFw1xLyLDw",
                     client_secret="ScPMZoWZOFA7Q7IMrEKtvBtvzUIp5w",
                     user_agent="praw_scraper_1.0"
                     )

# Create sub-reddit instance
subreddit_name = "programmingLanguages"
subreddit = reddit.subreddit(subreddit_name)

# looping over posts and scraping it
for post in subreddit.new(limit=None):
    comments = []
    polarity =
    language =
    post = Post(post,)