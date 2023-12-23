import streamlit as st
import pickle
import re
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')  

#loading models
clf1 = pickle.load(open('clf1.pkl','rb'))
# word2vec = pickle.load(open('word2vec.pkl','rb'))
word2vec = []
try:
    with open('word2vec.pkl', 'rb') as file:
        word2vec = pickle.load(file)
except FileNotFoundError:
    print("Error: 'word2vec.pkl' not found.")
except Exception as e:
    print(f"An error occurred while loading 'word2vec.pkl': {e}")


def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app

def get_average_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return None
    average_vector = sum(vectors) / len(vectors)
    return average_vector


def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')


        cleaned_resume = clean_resume(resume_text)
        input_features = get_average_vector(cleaned_resume, word2vec)
        input_features = input_features[:300]
        input_features = input_features.reshape(1, -1)
        prediction_id = clf1.predict(input_features)[0]
        st.write(prediction_id)
      
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Prediction ID:",prediction_id)
        st.write("Predicted Category:", category_name)



# python main
if __name__ == "__main__":
    main()
