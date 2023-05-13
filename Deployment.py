
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import en_core_web_lg
import spacy
import re
import string
import nltk
import pkg_resources
pkg_resources.require("googletrans==3.1.0a0")
import googletrans
from googletrans import Translator
import pickle

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------------------------------------------------------------------------------------------

# Set the page configuration
st.set_page_config(page_title="News Recommender App", page_icon=":guardsman:", 
                   layout="wide", initial_sidebar_state="expanded")

# Define the theme settings
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    color: #333333;
    background-color: #483D8B;
}
h1 {
    font-size: 48px;
    color: #0072b2;
}
h2 {
    font-size: 36px;
    color: #009e73;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------------- #
# defining a fuction to remove punctuations
def remove_punc(news_text):
    result = re.sub(r'[0-9]+', '', news_text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub('', result)
    return text

# defining a function to remove stop words
def remove_stop(the_text):
  # Tokenize the text into words
  words = word_tokenize(the_text)

  # Filter out the stop words from the text
  filtered_words = [word for word in words if not word in stop_words]

  # Join the filtered words into a string
  filtered_text = ' '.join(filtered_words)

  # return the filtered text
  return(filtered_text)

# prepare spacy model for lemmatization
nlp = spacy.load('en_core_web_lg')
nlp = en_core_web_lg.load()


# defining lemmatization function
def lemm(text):
    lemme=[]
    for token in nlp(text):
        lemme.append(token.lemma_)

    return " ".join(lemme)


# defining a fuction that combines all preprocessing steps needed
def preprocessing_text(text):
    text = remove_punc(text)
    text = remove_stop(text)
    text = lemm(text)
    return text

# ------------------------------------------------------------------------------------------------------ #

def main():
    
    # Adding the caching decorator
    @st.cache_data  
    def load_data():
        df = pd.read_csv('app_data.csv')
        return df


    # Load news articles data from CSV file
    news_df = load_data()


    # Define the title and text of the web app
    st.title("News Recommender App")
    st.subheader("Enter an article to find similar news articles")

    # Set up the translator
    translator = Translator()

    # Define the input field for the article text
    input_text = st.text_input("Article text")

    # Translate the text
    process = preprocessing_text(input_text)

    # translated text preprocessing
    try:
        translation = translator.translate(process , src='ar', dest='en')
        article_text = translation.text
    except:
        pass
# ________________________________________________________________________________________________________________ #
    # Define a button to recommend similar articles
    try:
        if st.button("Recommend"):
            # Initialize a TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer( max_features=5000, ngram_range=(1, 3) )

            # Fit and transform the news article texts
            tfidf_matrix = tfidf_vectorizer.fit_transform(news_df['new_body'])
            
            # Transform the input article text into a TF-IDF vector
            article_tfidf = tfidf_vectorizer.transform([article_text])

            # Compute the cosine similarity between the input article and all other articles
            cosine_similarities = cosine_similarity(article_tfidf, tfidf_matrix)

            # Get the index of the most similar article
            similar_article_index = cosine_similarities.argmax()

            # Get the titles of the top 5 recommended articles

            recommended_article_indices = cosine_similarity(tfidf_matrix[similar_article_index], tfidf_matrix).argsort()[0][-6:-1][::-1]
            # XGB classifier
            with open('trained_model.pkl', 'rb') as f:
                ml = pickle.load(f)
            
            
            categ = ml.predict( article_tfidf )[0]
            catgs = {"0":"محافظات" ,
                     "1" : "رياضة",
                     "2" : "عرب و عالم",
                     "3": "أخبار" ,
                     "4" : "ثقافة وفنون",
                     "5" : "حوادث",
                     "6" : "إقتصاد"}
            
            
            # Print the recommended articles
            st.header("Recommended articles:")
            
            
            for order,indx in list(enumerate(recommended_article_indices)):
                st.write(order+1 , "- " + news_df.iloc[indx]['headline'])
                st.write("Brief:     ", news_df.iloc[indx]['briefing'])
                st.write("Go to: ", news_df.iloc[indx]['hrefs'])
                st.write(" -----/----/-----/-----/-----/-----/------/-----/-----/-----/------ ")

            st.write("Classification of News: ", catgs[str(categ)])
    except:
        st.subheader("Please enter the news text")
    
    
if __name__ == '__main__':
    main()
