# new
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
# import numpy as np

# Step 1: Installing required libraries
# pip install transformers
# pip install torch

# sentences examples
sentences = ["Surely if you wanna make lots of $ then you should learn COBOL",
             "If you're interested in learning programming/seeing how it works/see if it's for you, Python's cool..",
             "Python is really bad",
             'I took two years of Java and wanted to kill myself',
             'I prefer Javascript over Python.',
             'If you wanna eventually learn something else, Java. If you want to do the same thing for all eternity, Python.',
             'As someone trying to learn c++ from python, I hate that I started with python.',
             'If someone ask me, just go with Python. Versatile enough for most application.',
             'C# but that is my fanboy/addict self talking',
             'I think you should start with python to learn how programs work, then maybe move to C/C++',
             'C# will make you love programming compared to others suggested here',
             'Python is such a versatile and easy-to-learn language. I love how it simplifies complex tasks.',
             "Java has been my go-to programming language for years. It's reliable and performs well for large projects.",
             "I find JavaScript to be a bit challenging at first, but once you get the hang of it, it's powerful for web development.",
             "C++ gives me great control over memory and performance, but it can be tricky to manage pointers.",
             "Ruby is a pleasure to work with. Its clean syntax makes coding fun and productive.",
             "I'm not a big fan of PHP. It feels outdated, and there are better alternatives for web development.",
             "Go is lightning-fast and perfect for building scalable systems. I'm impressed with its performance.",
             "Swift has been a game-changer for iOS development. It's modern and makes coding enjoyable.",
             "I had a bad experience with Perl. Its syntax is confusing, and debugging can be a nightmare.",
             "Rust has become my new favorite language. Its safety features and performance are top-notch."
             ]

# data
data = [['SENTENCE', "SENTIMENT"], ['________', "_________"], ]

# Step 2: Preparing our dataset (mock data)
train_text_list = [
    "Python's readability makes it a joy to work with.",
    "Debugging complex issues in Python can be challenging.",
    "JavaScript's versatility allows for both frontend and backend development.",
    "JavaScript's callback hell can make code difficult to maintain.",
    "Golang's performance and concurrency features are impressive.",
    "The learning curve of Golang can be steep for beginners.",
    "Ruby's elegant syntax promotes clean and readable code.",
    "Ruby's slower performance can be a concern for high-performance applications.",
    "Java's strong type system helps catch errors at compile time.",
    "Java's verbosity can make simple tasks require a lot of code.",
    "C# is well-integrated with the Microsoft ecosystem.",
    "The Windows-centric nature of C# can limit cross-platform development.",
    "Swift's safety features prevent many common programming errors.",
    "Swift's evolution can sometimes lead to breaking changes in newer versions.",
    "Rust's memory safety rules provide a strong foundation for secure programming.",
    "Rust's steep learning curve can be intimidating for newcomers.",
    "PHP's wide adoption ensures many resources and libraries are available.",
    "PHP's inconsistent function naming can be confusing.",
    "C++ offers high performance with fine-grained control over memory.",
    "C++'s complexity and potential for memory leaks can be daunting.",
    "TypeScript brings static typing to JavaScript, enhancing code quality.",
    "TypeScript's compilation step can slow down the development process.",
    "Haskell's strong type system prevents many runtime errors.",
    "Haskell's complex syntax and functional paradigm can be challenging.",
    "Perl's powerful text manipulation capabilities make it a favorite for scripting.",
    "Perl's 'write-only' code reputation can make maintenance difficult.",
    "Scala's mix of object-oriented and functional programming is powerful.",
    "Scala's complex syntax and steep learning curve can deter beginners.",
    "Kotlin's concise syntax improves productivity and code clarity.",
    "Kotlin's smaller community size might limit available resources.",
    "SQL's declarative nature simplifies database querying.",
    "Managing complex SQL queries can become convoluted over time.",
    "C's low-level control is useful for system programming.",
    "Memory management in C can lead to segmentation faults and memory leaks.",
    "Ruby on Rails accelerates web development with its conventions.",
    "Ruby on Rails can become less efficient for very complex applications.",
    "Assembly language provides direct control over hardware resources.",
    "Assembly's complexity makes it error-prone and time-consuming.",
    "PHP's low barrier to entry makes it accessible for beginners.",
    "PHP's inconsistent syntax and design choices can lead to confusion.",
    "Python's extensive libraries reduce the need for reinventing the wheel.",
    "Python's Global Interpreter Lock (GIL) limits multithreading performance.",
    "Java's platform independence allows for widespread deployment.",
    "Java's verbosity and boilerplate code can slow down development.",
    "C# has a rich set of tools for Windows application development.",
    "C#'s ecosystem is more limited when compared to Java's.",
    "JavaScript's asynchronous capabilities are crucial for web applications.",
    "JavaScript's type coercion can lead to unexpected behavior.",
    "Golang's simplicity and efficiency make it great for microservices.",
    "Golang's lack of generics can lead to code duplication.",
    "Swift's modern syntax makes it pleasant to write iOS apps.",
    "Swift's evolving nature can lead to incompatibility issues between versions.",
    "Rust's borrow checker enforces memory safety without a garbage collector.",
    "Rust's steep learning curve can be discouraging for newcomers.",
    "TypeScript's optional static typing improves code quality.",
    "TypeScript's extra layer of compilation can increase build times.",
    "Haskell's purity and strong typing prevent many bugs at compile time.",
    "Haskell's complex concepts and syntax can be overwhelming.",
    "Perl's regex capabilities make it a go-to language for text processing.",
    "Perl's readability can suffer due to its TMTOWTDI philosophy.",
    "Scala's functional programming features improve code maintainability.",
    "Scala's steep learning curve can be daunting for those new to functional programming.",
    "Kotlin's null safety features reduce the occurrence of null pointer exceptions.",
    "Kotlin's relative newness might mean fewer libraries compared to older languages.",
    "SQL's declarative syntax is intuitive for querying databases.",
    "Complex SQL queries can be challenging to optimize and maintain.",
    "C's performance and control are essential for system-level programming.",
    "Manual memory management in C can lead to difficult-to-debug errors.",
    "Ruby on Rails follows the convention over configuration principle.",
    "Ruby on Rails can become slower as applications grow in complexity.",
    "Assembly language allows for optimization of performance-critical tasks.",
    "Assembly's lack of abstraction can lead to error-prone code.",
    "PHP's dynamic typing simplifies coding but can lead to runtime errors.",
    "PHP's inconsistent function naming can be frustrating for developers.",
    "Python's simplicity and readability make it a great language for beginners.",
    "Python's Global Interpreter Lock (GIL) limits true parallelism.",
    "Java's vast ecosystem offers libraries and tools for various tasks.",
    "Java's boilerplate code can be verbose and repetitive.",
    "C# provides easy integration with Windows services and applications.",
    "C#'s ecosystem is more focused on Windows development.",
    "JavaScript's libraries and frameworks like React enable powerful web applications.",
    "JavaScript's loosely typed nature can result in unexpected behavior.",
    "Golang's fast compilation and runtime performance boost developer efficiency.",
    "Golang's lack of certain language features can lead to verbosity.",
    "Swift's performance and safety features make it suitable for iOS development.",
    "Swift's evolving language features can cause migration challenges.",
    "Rust's memory safety rules prevent common programming errors.",
    "Rust's steep learning curve can be discouraging for newcomers.",
    "TypeScript's static typing improves code quality and maintainability.",
    "TypeScript's additional layer of compilation can increase build times.",
    "Haskell's pure functional nature leads to robust and maintainable code.",
    "Haskell's steep learning curve can be a barrier to entry.",
    "Perl's text processing capabilities make it useful for scripting tasks.",
    "Perl's syntax can be less intuitive and harder to maintain over time.",
    "Scala's expressiveness allows for concise and powerful code.",
    "Scala's complex syntax can lead to confusion, especially for beginners.",
    "Kotlin's modern features and concise syntax enhance developer productivity.",
    "Kotlin's smaller community may result in fewer available resources.",
    "SQL's ability to work with relational databases is essential for many applications.",
    "Optimizing complex SQL queries can require deep understanding and effort.",
    "C's low-level control is crucial for building operating systems and hardware drivers.",
    "C's manual memory management can lead to bugs like buffer overflows.",
    "Ruby on Rails emphasizes convention over configuration for rapid development.",
    "Ruby on Rails can suffer from performance issues in certain situations.",
    "Assembly's direct control over hardware makes it essential for specific tasks.",
    "Writing code in assembly language is time-consuming and error-prone.",
    "PHP's widespread use ensures a large pool of available talent.",
    "PHP's inconsistent design choices can make code harder to read and maintain.",
]

train_labels = [
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0
]

# train_text_list = ["I love this product!","This movie is amazing.","The food was delicious.","I'm happy with the service.","The weather is perfect.","This book is terrible.","The customer support was terrible.","I hate this place.","The experience was awful.","This is the worst product ever.","I hate you","I do not like you","It is disgusting",]

# train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative

# val_text_list = [ "The hotel was nice.", "The staff was friendly.", "The coffee tasted bad.", "The app is user-friendly.","The design is outdated."]
val_text_list = ["Python's readability promotes clean and maintainable code.",
                 "Python's Global Interpreter Lock (GIL) can limit its performance.",
                 "Java's strong ecosystem offers tools for a wide range of domains.",
                 "Java's verbosity can make development slower compared to more modern languages.",
                 "C++ offers performance and control over memory for high-performance applications.",
                 "C++'s complex syntax can make it challenging to learn and use effectively.",
                 "Ruby's elegant syntax emphasizes developer happiness.",
                 "Ruby's slower performance can be a concern for performance-intensive applications.",
                 "PHP's easy entry allows beginners to quickly start coding.",
                 "PHP's inconsistent design choices can lead to less maintainable code.",
                 "SQL's querying capabilities are essential for database-driven applications.",
                 "Optimizing complex SQL queries can be time-consuming and require expertise.",
                 "C's low-level control is essential for systems programming.",
                 "C's manual memory management can lead to memory leaks and security vulnerabilities.",
                 "Assembly's direct hardware manipulation is crucial for certain tasks.",
                 "Assembly's complexity and lack of abstraction can make it error-prone.",
                 "Swift's performance and modern syntax make it a strong choice for iOS development.",
                 "Swift's rapid evolution can sometimes lead to challenges in keeping up with the changes.",
                 "TypeScript's static typing improves code quality and maintainability.",
                 "TypeScript's additional compilation step can increase build times.",
                 "Haskell's functional purity and static typing lead to reliable code.",
                 "Haskell's complex concepts and syntax can be challenging for newcomers.",
                 "Lua's lightweight design is suitable for embedding in applications.",
                 "Lua's relatively small community might limit available resources.",
                 "Elixir's focus on concurrency makes it a great choice for distributed systems.",
                 "Elixir's relatively small community might result in fewer libraries and resources.",
                 "Go's simple and concise syntax encourages writing clean code.",
                 "Go's lack of certain language features can lead to workarounds and verbosity.",
                 "Clojure's immutability and functional programming principles lead to reliable code.",
                 "Clojure's Lisp syntax can be unfamiliar and challenging for new developers.",
                 "Perl's regular expressions provide powerful text manipulation capabilities.",
                 "Perl's syntax can be difficult to read and maintain over time.",
                 "R's data analysis and visualization capabilities make it a popular choice.",
                 "R's performance can degrade with larger datasets and complex calculations.",
                 "Julia's high performance and expressive syntax make it a promising language.",
                 "Julia's smaller community might result in limited available libraries and resources.",
                 "Erlang's concurrency model enables highly scalable systems.",
                 "Erlang's syntax can be unusual and take some time to get used to.",
                 "Crystal's static typing and performance make it a compelling language.",
                 "Crystal's smaller community might mean fewer available libraries.",
                 "COBOL's legacy systems still play a critical role in various industries.",
                 "COBOL's verbosity and outdated constructs can make it challenging to work with.",
                 "Raku's focus on expressiveness results in code that's fun to write.",
                 "Raku's complexity and evolving features can be overwhelming for newcomers.",
                 "Hack's static type checking improves code quality in PHP applications.",
                 "Hack's smaller community size might limit resources compared to mainstream languages.",
                 "Fortran's historical significance continues to impact scientific computing.",
                 "Fortran's syntax and lack of modern features can be a barrier to new users.",
                 ]

# val_labels = [1, 1, 0, 1, 0]
val_labels = [
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0
]

# test_text_list = [ "The concert was fantastic!","The package arrived on time.","The interface is confusing.","The movie was a waste of time."]
test_text_list = ["Elixir's functional programming features make concurrent programming elegant.",
                  "Elixir's learning curve can be steep for developers new to functional programming.",
                  "Go's simplicity and efficiency are well-suited for building microservices.",
                  "Go's lack of generics can lead to code duplication in certain situations.",
                  "Lua's lightweight design makes it great for embedding in applications.",
                  "Lua's smaller ecosystem might limit available libraries for specific tasks.",
                  "Clojure's focus on immutability and concurrency is ideal for modern software.",
                  "Clojure's parentheses-heavy syntax can be intimidating for newcomers.",
                  "Perl's regular expressions make it a powerful tool for text manipulation.",
                  "Perl's lack of modern design can lead to difficult-to-read code.",
                  "R's data analysis capabilities make it a favorite among statisticians.",
                  "R's performance can become an issue with larger datasets.",
                  "Julia's high performance and expressive syntax are well-regarded.",
                  "Julia's smaller community may result in fewer resources compared to established languages.",
                  "Swift's safety features greatly reduce the risk of common programming errors.",
                  "Swift's evolving language can sometimes introduce compatibility issues.",
                  "Kotlin's modern syntax and features improve code readability.",
                  "Kotlin's adoption in the Android ecosystem ensures a strong developer community.",
                  "Scala's type inference simplifies code without sacrificing type safety.",
                  "Scala's complex syntax can be daunting for beginners.",
                  "Haskell's pure functional approach leads to robust and maintainable code.",
                  "Haskell's learning curve can be challenging for developers new to functional programming.",
                  "Rust's memory safety rules prevent many common bugs and vulnerabilities.",
                  "Rust's borrow checker can be restrictive and challenging to work with.",
                  "TypeScript's optional static typing helps catch errors before runtime.",
                  "TypeScript's additional compilation step can slow down the development process.",
                  "D's performance and safety features make it a strong language for system programming.",
                  "D's smaller community size might result in fewer libraries compared to mainstream languages.",
                  "Cobol's long history means it's still used in legacy systems.",
                  "Cobol's outdated syntax can be difficult for modern programmers to work with.",
                  "F# brings functional programming to the .NET ecosystem.",
                  "F#'s adoption rate might be lower due to its specific niche.",
                  "Perl's scripting capabilities enable automation and quick development.",
                  "Perl's reputation for unreadable code can hinder collaboration.",
                  "Rust's focus on memory safety is crucial for systems programming.",
                  "Rust's strict ownership rules can lead to complex code patterns.",
                  "C# provides excellent tooling for Windows application development.",
                  "C#'s ecosystem might not be as diverse as other programming languages.",
                  "JavaScript's dynamic nature allows for quick prototyping of ideas.",
                  "JavaScript's loosely typed nature can lead to unexpected errors.",
                  ]

# test_labels = [1, 1, 0, 0]
test_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0,
               1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
               ]

num_classes = 2  # Number of sentiment classes (positive and negative)

# Step 3: Loading Pre-trained BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 4: Data Preprocessing
encoded_data_train = tokenizer.batch_encode_plus(
    train_text_list,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    val_text_list,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    test_text_list,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_labels)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_labels)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(test_labels)

train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Step 6: Defining Sentiment Analysis Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Step 7: Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Enabling fine-tuning
model.train()

optimizer = optim.AdamW(model.parameters(), lr=2e-5)

num_epochs = 3

# Enabling fine-tuning
model.train()

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 10,
    num_training_steps=len(train_loader) * num_epochs
)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Step 8: Evaluation
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for batch in val_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy}")

# Step 9: Testing
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for batch in test_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy}")


def make_analysis(text_to_classify):
    encoded_text = tokenizer.encode_plus(
        text_to_classify,
        add_special_tokens=True,
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        row = [text_to_classify, predicted_class]
        data.append(row)
        print(f"Predicted Sentiment Class: {predicted_class}")


# Step 10: Inference
# text_to_classify = "I hate you
for sentence in sentences:
    make_analysis(sentence)

print(data, device)

""" def showGraph():
    # x-coordinates of left sides of bars
    left = np.arange(len(allData))

    # heights of bars
    heightPositive = []
    heightNegative = []

    # labels for bars
    tick_label = []

    # widths of the bars
    bar_width = 0.4

    for data in allData:
        heightPositive.append(len(allData[data]["positiveSentences"]))
        heightNegative.append(len(allData[data]["negativeSentences"]))
        tick_label.append(data)

    # plotting the bars for positive and negative sentiments side by side
    plt.bar(left, heightPositive, width=bar_width, label='Positive', color='green')
    plt.bar(left + bar_width, heightNegative, width=bar_width, label='Negative', color='red')

    # naming the x-axis
    plt.xlabel('Languages')
    # naming the y-axis
    plt.ylabel('Number of Sentences')
    # plot title
    plt.title('Positive and Negative Sentiments for Each Language')

    # setting the x-ticks to be at the middle of each group of bars
    plt.xticks(left + bar_width / 2, tick_label)

    # Rotate tick labels vertically
    plt.xticks(rotation='vertical')

    # displaying the legend
    plt.legend()

    # function to show the plot
    plt.show()
"""
# showGraph()
