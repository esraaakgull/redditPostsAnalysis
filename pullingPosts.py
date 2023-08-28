import praw
import pandas as pd

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

# creating dataframe for displaying scraped data
df = pd.DataFrame()

# creating lists for storing scraped data
titles = []
scores = []
text = []

# looping over posts and scraping it
for submission in subreddit.new(limit=10):
    titles.append(submission.title)
    scores.append(submission.score)  # upvotes
    text.append(submission.selftext)
    # her postun yorumları
   # for comment in submission.comments.list():
        # print(comment.body)

df['Title'] = titles
df['Text'] = text
# df['Upvotes'] = scores  # upvotes

print(df.shape)
print(df.head(10))
