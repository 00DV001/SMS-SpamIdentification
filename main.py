import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import joblib

# importing data
csv_path = "sms_scam_detection_dataset_merged_with_lang.csv"
df = pd.read_csv(csv_path)
#df_info = df.info()
df_preview = df.head()

# cleaning data
columns_to_drop = [col for col in df.columns if 'Unnamed' in col or col in ['URL', 'EMAIL', 'PHONE']]
df_cleaned = df.drop(columns=columns_to_drop)  # drop unnamed or unwanted cols
df_cleaned['label'] = df_cleaned['label'].str.lower()
print(df_cleaned['label'].value_counts())

df_cleaned = df_cleaned.dropna(subset=['text'])  # drop missing text
label_values = df_cleaned['label'].unique()  # unique value check
df_cleaned.reset_index(drop=True, inplace=True)  # indexing reset
# print(df_cleaned.shape, label_values)  # cleaned output

# unique labelling for spam
df_cleaned['label'] = df_cleaned['label'].str.lower()
label_counts = df_cleaned['label'].value_counts() # checking class distribution, print(label_counts)

def preprocess_text(text): # Basic text preprocessing function
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'\d{100,}', '', text)  # remove phone numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# applying to the text column
df_cleaned['clean_text'] = df_cleaned['text'].apply(preprocess_text)
# initializing TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 1),
    sublinear_tf=True,
    min_df=1,
    max_df=0.9
)

# fit and transform the clean_text column
X = vectorizer.fit_transform(df_cleaned['clean_text'])
# encode the labels: spam = 1, ham = 0
y = df_cleaned['label'].map({'ham': 0, 'spam': 1})

# show the shape of the result
print(X.shape, y.shape)

# split the data into train and test sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# show the shapes of the splits
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# initialize and train the model
model = MultinomialNB(alpha=2.0)
model.fit(X_train, y_train)

# predict on the test set
y_pred = model.predict(X_test)

# evaluate performance
print( f"MNB Accuracy: {accuracy_score(y_test, y_pred) * 100 : .2f} ")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# convert y_test to a Series (if not already)
y_test_series = y_test.reset_index(drop=True)
y_pred_series = pd.Series(y_pred)

# save the trained model
joblib.dump(model, "spam_classifier_model.pkl")
# save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
