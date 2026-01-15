import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from datetime import datetime

data = pd.read_csv("spam.csv")



data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])



mess = data['Message']
cat = data['Category']



(mess_train, mess_test, cat_train, cat_test) = train_test_split(
    mess, cat, test_size=0.2
)



cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)



# creating Model
model = MultinomialNB()
model.fit(features, cat_train)



# Test our model
features_test = cv.transform(mess_test)
# print(model.score(features_test, cat_test))



# predict Data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

# ===== STREAMLIT UI STARTS HERE =====

# Page configuration
st.set_page_config(
    page_title="Spam Detection",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Model Info")
    accuracy = model.score(features_test, cat_test)
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
    
    st.markdown("---")
    st.subheader("About")
    st.write("This app detects spam messages using Naive Bayes algorithm.")
    st.write("**How to use:**")
    st.write("1. Enter your message")
    st.write("2. Click Detect")
    st.write("3. See the result")

# Main content
st.title("🛡️ Spam Detection System")
st.write("Enter a message below to check if it's spam or not.")

st.markdown("---")

# Input
input_mess = st.text_area(
    "Your Message",
    placeholder="Enter message here...",
    height=120
)

# Detect button
if st.button("🔍 Detect Spam"):
    if input_mess.strip():
        output = predict(input_mess)
        
        st.markdown("---")
        st.subheader("Result:")
        
        if output[0] == 'Spam':
            st.error("🚫 This is SPAM!")
            st.warning("⚠️ This message appears to be spam. Be careful with suspicious links and requests.")
        else:
            st.success("✅ This is NOT SPAM")
            st.info("ℹ️ This message looks safe, but always verify the sender.")
    else:
        st.warning("Please enter a message to analyze.")

# Examples
st.markdown("---")
st.subheader("Examples to try:")

col1, col2 = st.columns(2)
with col1:
    st.write("**Spam:**")
    st.text("• Win $1000 now!\n• Click here urgently\n• Free prize claim")
with col2:
    st.write("**Not Spam:**")
    st.text("• Meeting at 3pm\n• Thanks for email\n• Project update")

st.markdown("---")
st.caption("Built with Streamlit & Machine Learning")