import spacy
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the Articles.csv file into a pandas dataframe
df = pd.read_csv("archive/Articles.csv", encoding="Windows-1252")

# Remove leading and trailing white spaces in the columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

# Remove rows with invalid dates
df = df[df["Date"].apply(lambda x: bool(re.match(r"\d{1,2}/\d{1,2}/\d{4}", x)))]

# Remove rows with invalid characters in the "Article" column
df = df[df["Article"].apply(lambda x: bool(re.match(r"^[a-zA-Z0-9.,:;!?\'\"\s]+$", x)))]

# Remove rows with invalid characters in the "Heading" column
df = df[df["Heading"].apply(lambda x: bool(re.match(r"^[a-zA-Z0-9.,:;!?\'\"\s]+$", x)))]

# Remove rows with invalid characters in the "NewsType" column
df = df[
    df["NewsType"].apply(lambda x: bool(re.match(r"^[a-zA-Z0-9.,:;!?\'\"\s]+$", x)))
]

# Save the cleaned dataframe to a new CSV file
df.to_csv("archive/cleaned_articles.csv", index=False)

# Create a spaCy nlp object to process the text data in the dataframe
nlp = spacy.load("en_core_web_sm")

# Define a function to analyze the sentiment of each article
def analyze_sentiment(text):
    print(text)
    doc = nlp(text)
    return doc.sentiment.polarity


# Apply the sentiment analysis function to each article in the dataframe
df["sentiment"] = df["Article"].apply(analyze_sentiment)

# Use matplotlib to create a histogram of the sentiment scores
plt.hist(df["sentiment"], bins=10)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Distribution of Sentiment in Articles")
plt.savefig("sentiment_histogram.png")
plt.show()
