import time
import praw
import csv

# field names
fields = ['Number', 'Title', 'Content', 'Comments']

# name of csv file
filename = "programmingLanguages.csv"

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

# Define rate limiting parameters
REQUESTS_PER_MINUTE = 60  # Adjust this based on Reddit's API rate limits
RATE_LIMIT_SLEEP_SECONDS = 60 / REQUESTS_PER_MINUTE

# writing to csv file
with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    num = 1
    # looping over posts and scraping it
    for submission in subreddit.new(limit=None):
        title = submission.title
        text = submission.selftext
        comments = []
        for comment in submission.comments.list():
            if isinstance(comment, praw.models.Comment):
                comments.append(comment.body.replace('\n', ' ').replace('\r', ''))

        # Join comments into a single string with a delimiter
        comments_combined = '; '.join(comments)

        csvwriter.writerow([num, title, text, comments_combined])
        num = num + 1

        # Implement rate limiting
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)  # Pause to avoid rate limit issues
