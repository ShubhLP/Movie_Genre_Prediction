import streamlit as st
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize session state if needed
if 'model' not in st.session_state:
    try:
        # Define the model architecture
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)
        
        # Load the state dictionary
        model.load_state_dict(torch.load('movie_genre_classifier.pth', map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode

        # Store in session state
        st.session_state.model = model

    except Exception as e:
        st.error(f"Error loading model: {e}")

if 'tokenizer' not in st.session_state:
    try:
        # Load the tokenizer
        st.session_state.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")

if 'mlb' not in st.session_state:
    try:
        # Load MultiLabelBinarizer
        with open('mlb.pkl', 'rb') as f:
            st.session_state.mlb = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading MultiLabelBinarizer: {e}")

# Define the prediction function
def predict_genres(synopsis, threshold=0.5):
    try:
        # Tokenize the input
        inputs = st.session_state.tokenizer(synopsis, 
                                          return_tensors='pt', 
                                          truncation=True, 
                                          padding=True, 
                                          max_length=256)
        
        # Ensure inputs are on CPU
        inputs = {k: v.to(torch.device('cpu')) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = st.session_state.model(**inputs)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        # Convert predictions to genres based on threshold
        predicted_genres = st.session_state.mlb.inverse_transform(
            (predictions > threshold).astype(int).reshape(1, -1))[0]
        
        return list(predicted_genres)
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return []

# Create the Streamlit app
st.title("Movie Genre Prediction")
st.write("Enter a movie synopsis to get its predicted genres.")

# Add a text area for input
synopsis = st.text_area("Movie Synopsis", "", height=200)

# Add a prediction button
if st.button("Predict"):
    if synopsis:
        try:
            with st.spinner('Predicting genres...'):
                predicted_genres = predict_genres(synopsis)
                if predicted_genres:
                    st.success("Prediction completed!")
                    st.write("Predicted Genres:", ", ".join(predicted_genres))
                else:
                    st.warning("No genres were predicted. Try a different synopsis.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Please enter a synopsis before predicting.")

# Add some information about the app
st.markdown("""
---
### About this app
This app uses a BERT-based model to predict movie genres based on their synopsis. 
The model has been trained on a dataset of movie synopses and their corresponding genres.
""")
