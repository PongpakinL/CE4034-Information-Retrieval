import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('data.csv')

# Create a corpus of the keywords
corpus = df['keyword'].to_numpy()

# Create a TF-IDF vectorizer and extract features
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(corpus)

# Get the search query from the user
search_query = input('Enter your search query: ')

# Convert the search query to TF-IDF representation
query_tfidf = vectorizer.transform([search_query])

# Compute the cosine similarities between the search query and the documents
cosine_similarities = cosine_similarity(query_tfidf, features)[0]

# Set the cosine similarity threshold
cosine_threshold = 0.2

# Get the indices of the documents that have cosine similarity greater than the threshold
related_indices = np.where(cosine_similarities > cosine_threshold)[0]

# Sort the related indices in descending order of cosine similarity
sorted_indices = np.argsort(cosine_similarities[related_indices])[::-1]

# Check if there are any matching documents
if len(sorted_indices) == 0:
    print("Sorry, no matching documents found.")
else:
    # Print the number of matching documents found
    print(f"Found {len(sorted_indices)} matching documents:")
    
    # Loop through the sorted indices and print the corresponding documents and labels
    for i in sorted_indices:
        print(df['text'][related_indices[i]])
        print(df['label'][related_indices[i]])
        print()
