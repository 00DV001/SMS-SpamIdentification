# SMS-SpamIdentification
SMS Spam Detection App:
A machine learning-powered web app built using Streamlit to detect spam (scam) SMS messages. The app uses Multinomial Naive Bayes with TF-IDF vectorization to classify text messages as either Spam or Ham (Not Spam).

Features:
1. Clean and user-friendly Streamlit interface
2. Predict if a message is spam or not spam
3. Preprocessing and vectorization using TF-IDF (unigrams & bigrams)
4. Model: Multinomial Naive Bayes 
5. Accurate classification with over 83% accuracy 

Tech Stack:
1. Python 3.x
2. Pandas, Scikit-learn
3. Streamlit
4. Regex for text preprocessing
5. TF-IDF Vectorizer
6. MultinomialNB classifier

Dataset: 
The app uses a merged dataset of SMS messages with labels spam and ham. Example of a message:
  Congratulations! You've won a free iPhone! Click here to claim: http://spamlink.com

Installation & Setup:
1. Clone the repository:
  git clone [https://github.com/your-username/spam-detector-app.git](https://github.com/00DV001/SMS-SpamIdentification.git);
  cd spam-detector-app;
2. Create a virtual environment (optional but recommended)
	python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies
  pip install -r requirements.txt
4. Run the app
  streamlit run app.py

How It Works
1. Loads and cleans SMS text messages
2. Preprocesses text (lowercase, punctuation removal, etc.)
3. Transforms text using TF-IDF vectorizer (1-2 grams)
4. Classifies using Multinomial Naive Bayes
5. Displays prediction for the input message

Model Performance:
1. Metric      Score
2. Accuracy    83%
3. Precision   0.89 (ham), 0.77 (spam)
4. Recall      0.80 (ham), 0.87 (spam)

File Structure
1. app.py                  # Main Streamlit app
2. sms_scam_dataset.csv    # Your labeled SMS dataset
3. requirements.txt        # Python dependencies
4. README.md               # Project documentation

Contact
1. Made by Devansh Vats
2. GitHub: @00DV001
