import praw
from nltk import word_tokenize

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
for post in subreddit.new(limit=100):
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


make_post_content_analysis()
make_post_comment_analysis()
# print(languagePosts)
print(languageComments)

# post başlığı: post.title
# post texti: post.selftext
# post yorumları: post.comment.list()
# yorum texti: comment.body

# her postun yorumları
# for comment in submission.comments.list():
# print(comment.body)
