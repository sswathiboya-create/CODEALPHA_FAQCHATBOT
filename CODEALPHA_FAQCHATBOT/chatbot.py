from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# FAQs
questions = [
    "What is AI?",
    "What is Machine Learning?",
    "What is Python?",
    "What is Deep Learning?"
]

answers = [
    "AI is the simulation of human intelligence by machines.",
    "Machine Learning is a subset of AI.",
    "Python is a programming language.",
    "Deep Learning is a part of Machine Learning using neural networks."
]

# Convert text into vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(questions)

# Chatbot function
def chatbot(user_question):

    user_vec = vectorizer.transform([user_question])

    similarity = cosine_similarity(user_vec, X)

    index = np.argmax(similarity)

    # Confidence check
    if similarity[0][index] < 0.3:
        return "Sorry, I don't understand."

    return answers[index]

# Chat Loop
print("FAQ Chatbot (Type 'exit' to stop)")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = chatbot(user_input)

    print("Chatbot:", response)