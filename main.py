import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from wordcloud import WordCloud

# Download stopwords
nltk.download('stopwords')

# Step 1: Load the dataset
twitter_df = pd.read_csv("twitter_data.csv")

# ✅ Manually set columns based on your dataset
text_column = 'clean_text'
sentiment_column = 'category'

# Step 2: Clean the text data (only if needed)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.split()
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Only clean if the text isn't already preprocessed
if 'cleaned_text' not in twitter_df.columns:
    twitter_df['cleaned_text'] = twitter_df[text_column].apply(clean_text)
else:
    twitter_df['cleaned_text'] = twitter_df[text_column]

# Step 3: Clean the text data (only if needed)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean text (already done above)
twitter_df['cleaned_text'] = twitter_df[text_column]

# ✅ Drop missing values in text and category columns
twitter_df = twitter_df.dropna(subset=['cleaned_text', sentiment_column])

# ✅ Convert labels to string (optional but good)
twitter_df[sentiment_column] = twitter_df[sentiment_column].astype(str)

# ✅ TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(twitter_df['cleaned_text'])
y = twitter_df[sentiment_column]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Split and Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Visualize Sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=sentiment_column, data=twitter_df)
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.show()

# Step 7: WordCloud for Positive Sentiment (if applicable)
if 'positive' in twitter_df[sentiment_column].unique():
    positive_text = " ".join(twitter_df[twitter_df[sentiment_column] == 'positive']['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400).generate(positive_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Positive Sentiment WordCloud")
    plt.show()
