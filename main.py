import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()

import streamlit as st


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# step2
@st.cache_data
def clean_data(raw_text):
        clean_text = re.sub(r'\/[A-Za-z\-]+', '', raw_text)
        clean_text = re.sub(r'[\/\.\n\t]', '', clean_text)
        clean_text = re.sub(r'`', '', clean_text)
        return clean_text
@st.cache_data
def preprocess(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]  # Remove non-alphanumeric characters
    words = [word for word in words if word.lower() not in stop_words]  # Remove stop words
    words = [ps.stem(word) for word in words]  # Apply stemming
    return words
# indexterm dictinary

index_term_dict = {}
# import nltk
corpus_root = './newcorpus'
corpus_reader = nltk.corpus.PlaintextCorpusReader(corpus_root, '.*\.txt')

fileids = corpus_reader.fileids()

for i in range(0, 6):

    raw_text = corpus_reader.raw(fileids[i])
    clean_text = clean_data(raw_text)
    text2 = preprocess(clean_text)
    for t in text2:
      if t in index_term_dict:
            if not isinstance(index_term_dict[t], list):

                index_term_dict[t] = [index_term_dict[t]]

            index_term_dict[t].append(fileids[i])

      else:

            index_term_dict[t] = [fileids[i]]

print(index_term_dict)


def preprocess(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]  # Remove non-alphanumeric characters
    words = [word for word in words if word.lower() not in stop_words]  # Remove stop words
    words = [ps.stem(word) for word in words]  # Apply stemming
    return words




#UI LOGIC 
import streamlit as st

# Set the title of the app
st.title("Question and Answer App")

# Create a container for the frame in the middle
with st.container():
    st.write("### Ask a Question")
    
    # Input field for the question
    question = st.text_input("Your Question:")
    st.write("*Note: Please ensure your question is related to the domain(Children Act).*")
    # question="periodical payments order"
    # Button to submit the question
    if st.button("Ask"):
        if question:
            # Display the answer field with a placeholder response
            processed_query = preprocess(question)
             #document retrival
            def retrieve_documents(processed_query):
                doc_sets = []
                for word in processed_query:
                    if word in index_term_dict:
                        doc_sets.append(set(index_term_dict[word]))

                # Find intersection of all document sets
                if doc_sets:
                    common_docs = set.intersection(*doc_sets)
                else:
                    common_docs = set()

                return common_docs
                # return doc_sets

            # Example
            documents = retrieve_documents(processed_query)


            #keyword ranking and jaccard similarity
            def jaccard_similarity(query_set, doc_set):
                intersection = query_set.intersection(doc_set)
                union = query_set.union(doc_set)
                return len(intersection) / len(union)

            def rank_documents(processed_query, documents):
                query_set = set(processed_query)
                ranked_docs = []

                for doc in documents:
                    with open(f'./newcorpus/{doc}', 'r',encoding='utf-8') as file:
                        doc_text = file.read()
                        doc_words = set(preprocess(doc_text))
                        score = jaccard_similarity(query_set, doc_words)
                        ranked_docs.append((doc, score))

                ranked_docs.sort(key=lambda x: x[1], reverse=True)
                return ranked_docs

            # Example
            ranked_documents = rank_documents(processed_query, documents)
            print(ranked_documents)
            if len(ranked_documents) == 0:
                    st.error("No documents available.")
        # Stop further execution
                    # return
            else:
                    
                    def extract_relevant_sentences(document, query, top_n=2):
                    # Split the document into sentences
                        sentences = document.split('. ')
        
                        # Append query to the list of sentences
                        sentences.append(query)
        
                        # Vectorize the sentences and the query using TF-IDF
                        vectorizer = TfidfVectorizer().fit_transform(sentences)
        
                        # Compute cosine similarity between the query and all sentences
                        cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1])
        
                        # Flatten the cosine similarities array and get indices of top_n sentences
                        similarities = cosine_similarities.flatten()
                        relevant_indices = similarities.argsort()[-top_n:][::-1]
        
                        # Extract the top_n relevant sentences
                        relevant_sentences = [sentences[i] for i in relevant_indices]
        
                        return relevant_sentences
        
                    # Example usage
                    doc=ranked_documents[0][0]
                    with open(f'./newcorpus/{doc}', 'r',encoding='utf-8') as file:
                        doc_text = file.read()
                    document=doc_text
                    # query = "DOES CHATGPT PROTECT PRIVACY"
                    formatted_sentences = []
                    # Extract and print only the first two relevant sentences
                    relevant_sentences = extract_relevant_sentences(document, question, top_n=1)
                    for idx, sentence in enumerate(relevant_sentences, 1):
                        print(f"{idx}. {sentence}")
                        formatted_sentence = f"{idx}. {sentence}"
                        formatted_sentences.append(formatted_sentence)
                    formatted_sentences_str = "\n".join(formatted_sentences)
        
                    if len(formatted_sentences_str.strip()) == 0:
                        # st.error("Please enter a valid question.")
                        st.text_area("Answer", value="Please enter a valid question.", height=300, disabled=True)
                    else:
                        st.text_area("Answer", value=formatted_sentences_str, height=300, disabled=True)
                    print("formatted sentence-----------------------------------------------------------------")
                    print(formatted_sentences)
                    # answer=formatted_sentence[2]
        
        
        
        
                    
        
                    # answer = f"This is a placeholder answer for your question: '{question}'"
                    # st.text_area("Answer:", value=answer, height=200)
                else:
                    st.error("Please enter a question.")
        
        # Run the app using the command: streamlit run app.py







#user Question
# question="periodical payments order"


# stop_words = set(stopwords.words('english'))
# ps = PorterStemmer()



# Example of processing a text file
# with open('corpus', 'r') as file:
#     text = file.read()
#     processed_text = preprocess(text)
#     print(processed_text)



# print(documents)     



# Answer Extraction



