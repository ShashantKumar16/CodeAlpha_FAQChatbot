import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Custom tokenizer (avoids word_tokenize)
def simple_tokenize(text):
    text = text.lower()
    tokens = text.translate(str.maketrans('', '', string.punctuation)).split()
    return [word for word in tokens if word not in stopwords.words('english')]

# Preprocess function
def preprocess(text):
    return ' '.join(simple_tokenize(text))

# Download stopwords (only once)
nltk.download('stopwords')

# Sample FAQ data
faq_data = {
    "What is your return policy?": "You can return items within 30 days of purchase.",
    "How can I contact customer support?": "You can contact us at support@example.com.",
    "Do you ship internationally?": "Yes, we ship to most countries worldwide.",
    "What payment methods are accepted?": "We accept credit cards, PayPal, and UPI.",
    "How do I track my order?": "You can track your order using the tracking link sent to your email.",
    "How can I reset my password?": "Click on 'Forgot Password' at the login screen to reset it.",
}

# Prepare data
questions = list(faq_data.keys())
preprocessed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

# Streamlit UI
st.title("🤖 FAQ Chatbot with Custom NLP - CodeAlpha")
st.markdown("Ask a question and get the most relevant answer using Python-based NLP.")

user_input = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        cleaned_input = preprocess(user_input)
        user_vector = vectorizer.transform([cleaned_input])
        similarities = cosine_similarity(user_vector, X)
        most_similar_index = similarities.argmax()
        best_match = questions[most_similar_index]
        st.success(f"**Answer:** {faq_data[best_match]}")
