import google.generativeai as genai
import streamlit as st

GOOGLE_API_KEY='AIzaSyDB4dGbIs4oMKUjd9aTXXIuQOZW3HuPRt4'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-pro-latest')

def check_sentiment(feedback):
    prompt = f"""You are a sentiment classification model. Refer below exampleFeedback: {feedback}\nSentiment: Only return 'p' if positive or 'n' if negative.
        ###
        feedback : What a lovely product
        sentiment : p
        ###
        feedback : I am not happy with this product
        sentiment : n
        ###
        feedback : {feedback}
        sentiment :
        """
    response = model.generate_content(
            prompt
        )
    sentiment = response.text.strip()
    return sentiment

st.title("Product Feedback Sentiment Analysis")
st.write("Enter your product feedback below:")

feedback = st.text_area("Product Feedback", height=150)

if st.button("Analyze Sentiment"):
    if feedback.strip() == "":
        st.error("Please enter feedback before submitting.")
    else:
        sentiment = check_sentiment(feedback)
        print(sentiment)
        if 'p' in sentiment:
            st.success(sentiment) 
        else:
            st.warning(sentiment)