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
    page_title="Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    .title-text {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ Spam Detector")
    st.markdown("---")
    
    # Model stats
    accuracy = model.score(features_test, cat_test)
    st.metric("🎯 Model Accuracy", f"{accuracy*100:.2f}%")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info(f"""
    **Total Messages:** {len(data):,}  
    **Spam:** {len(data[data['Category']=='Spam']):,}  
    **Not Spam:** {len(data[data['Category']=='Not Spam']):,}
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write("""
    This AI-powered system uses **Naive Bayes** algorithm to detect spam messages.
    
    **How to use:**
    1. Enter your message
    2. Click Analyze
    3. Get instant results
    """)
    
    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Main content
st.markdown('<h1 class="title-text">🛡️ Spam Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Message Classification | Protect Yourself from Spam</p>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("---")
    
    # Input section with better styling
    st.markdown("### 📝 Enter Your Message")
    input_mess = st.text_area(
        "Message",
        placeholder="Type or paste your message here to check if it's spam...\n\nExample: 'Congratulations! You've won $1000. Click here to claim now!'",
        height=150,
        label_visibility="collapsed"
    )
    
    # Buttons in columns
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    
    with btn_col2:
        analyze_button = st.button("🔍 Analyze Message", use_container_width=True)
    
    if analyze_button:
        if input_mess.strip():
            with st.spinner("🔄 Analyzing..."):
                output = predict(input_mess)
                
                st.markdown("---")
                st.markdown("### 📋 Detection Result")
                
                # Display result with enhanced styling
                if output[0] == 'Spam':
                    st.error("### 🚫 SPAM DETECTED!")
                    st.markdown("""
                    <div style='padding: 20px; background-color: #ffe6e6; border-left: 5px solid #ff4444; border-radius: 5px;'>
                        <h4 style='color: #cc0000; margin: 0;'>⚠️ Warning: This message appears to be spam</h4>
                        <p style='margin: 10px 0 0 0; color: #666;'>
                        Common spam indicators may include: promotional content, suspicious links, 
                        urgency tactics, or requests for personal information.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
                    
                else:
                    st.success("### ✅ SAFE MESSAGE")
                    st.markdown("""
                    <div style='padding: 20px; background-color: #e6ffe6; border-left: 5px solid #44ff44; border-radius: 5px;'>
                        <h4 style='color: #009900; margin: 0;'>✓ This message appears to be legitimate</h4>
                        <p style='margin: 10px 0 0 0; color: #666;'>
                        No spam indicators detected. However, always exercise caution with 
                        messages from unknown sources.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional info
                st.markdown("---")
                st.info("💡 **Tip:** Always verify sender identity and avoid clicking suspicious links, even if a message is classified as safe.")
                
        else:
            st.warning("⚠️ Please enter a message to analyze.")
    
    # Example messages section
    st.markdown("---")
    st.markdown("### 💭 Try These Examples")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.markdown("""
        **🔴 Spam Examples:**
        - "WINNER! You've been selected..."
        - "Click here to claim your prize!"
        - "Urgent: Your account will be closed"
        """)
    
    with example_col2:
        st.markdown("""
        **🟢 Safe Examples:**
        - "Meeting scheduled for tomorrow"
        - "Thanks for your email"
        - "Project update attached"
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #888;'>
        <p style='margin: 0;'>Built with ❤️ using Streamlit & Machine Learning</p>
        <p style='margin: 5px 0 0 0; font-size: 14px;'>© 2026 Spam Detection System</p>
    </div>
""", unsafe_allow_html=True)