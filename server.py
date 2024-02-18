from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import time

app = Flask(__name__)


df = pd.read_csv('C:\\Personal\\Nanyang Technological University\\Year 3\\Semester 2\\CE4034 Information Retrieval\\Course Project\\data.csv')
corpus = df['keyword'].to_numpy()
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(corpus)
cosine_threshold = 0

def count_most_frequent_words(df, n=20):
    words = ' '.join(df).split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(n)
    word_cloud_data = [{'text': word, 'weight': count} for word, count in most_common_words]
    return word_cloud_data

def query(search, filter):
    start_time = time.time()
    query_tfidf = vectorizer.transform([search])
    cosine_similarities = cosine_similarity(query_tfidf, features)[0]
    related_indices = np.where(cosine_similarities > cosine_threshold)[0]
    sorted_indices = np.argsort(cosine_similarities[related_indices])[::-1]
    sorted_related = related_indices[sorted_indices]
    if filter == 'Positive':
        sorted_related = sorted_related[df['label'].values[sorted_related] == 1]
    elif filter == 'Negative':
        sorted_related = sorted_related[df['label'].values[sorted_related] == -1]
    elif filter == 'Neutral':
        sorted_related = sorted_related[df['label'].values[sorted_related] == 0]
    
    result = []
    pie = [(df['label'][sorted_related] == 1).sum(),
             (df['label'][sorted_related] == -1).sum(),
             (df['label'][sorted_related] == 0).sum()]
    for i in sorted_related:
        result.append([df['text'][i], int(df['label'][i])])
    word_cloud_data = count_most_frequent_words(df['keyword'][sorted_related], n=20)
    end_time = time.time()
    print(end_time - start_time)
    return [result, pie, word_cloud_data]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    search = request.form['search']
    filter = request.form.get('filter')
    result = query(search, filter)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)