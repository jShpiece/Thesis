import os
import fitz  # PyMuPDF for handling PDF files
import numpy as np
import re
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
import nltk
nltk.download('stopwords')

def extract_text_from_pdf(file_path):
    '''
    Function to extract text from a PDF file
    :param file_path: Path to the PDF file
    :return: Extracted text from the PDF file
    '''
    text = ''
    try:
        document = fitz.open(file_path)
        for page in document:
            try:
                # Attempt to extract text from the page
                text += page.get_text("text")
            except Exception as e:
                print(f"Error extracting text on page of {file_path}: {e}")
        document.close()
    except Exception as e:
        print(f'Error reading {file_path}: {e}')
        text = None  # Handle other generic errors
    return text

def clean_text(text):
    '''
    Preprocess the text to remove stop words, numbers, and unnecessary characters
    '''
    # Remove digits and special characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return filtered_text

def read_first_chapter(book_path):
    '''
    Function to read the first 'chapter' of a textbook
    :param book_path: Path to the textbook PDF file
    :return: First 'chapter' of the textbook (approximately the first 100,000 characters)
    '''
    first_chapter = extract_text_from_pdf(book_path)

    if first_chapter and first_chapter.strip():  # Ensure the chapter is not empty
        # Preprocess the extracted text
        cleaned_text = clean_text(first_chapter[:1000000000])  # Clean and limit to 1,000,000,000 characters
        return cleaned_text
    else:
        print(f"Skipped {book_path} because its first chapter is empty or invalid.")
        return None  # Return None if the PDF couldn't be read or is empty

def find_optimal_clusters(embeddings, max_k=20):
    '''
    Function to calculate silhouette scores and find the optimal number of clusters
    :param embeddings: Array of text embeddings
    :param max_k: Maximum number of clusters to try
    :return: Optimal number of clusters based on silhouette score
    '''
    silhouette_scores = []
    for n_clusters in range(2, 20):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append((n_clusters, score))
        print(f"Number of clusters: {n_clusters}, Silhouette Score: {score}")
    
    # Choose the best number of clusters based on the highest silhouette score
    best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    print(f"Optimal number of clusters: {best_n_clusters}")
    return best_n_clusters

def organize_textbooks_by_similarity(textbook_dir, max_k=20):
    '''
    Function to organize textbooks by similarity of their first chapters using sentence embeddings
    :param textbook_dir: Directory containing the textbooks
    :param max_k: Maximum number of clusters to try for silhouette scoring
    :return: Dictionary where keys are cluster labels and values are lists of textbooks in each cluster
    '''
    # Initialize sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Log how many textbooks are detected in the directory
    textbooks = os.listdir(textbook_dir)
    print(f"Total files found: {len(textbooks)}")
    
    first_chapters = {}
    embeddings = []

    # Read and preprocess first chapters
    for textbook in textbooks:
        book_path = os.path.join(textbook_dir, textbook)
        # first_chapter = read_first_chapter(book_path)
        first_chapter = extract_text_from_pdf(book_path) # Lets use the entire text for now
        if first_chapter is not None:
            # print(f"Successfully processed text for '{textbook}'")
            first_chapters[textbook] = first_chapter
            try:
                embedding = model.encode(first_chapter)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    print(f"Embedding for '{textbook}' is None. Skipping.")
            except Exception as e:
                print(f"Error encoding textbook '{textbook}': {e}")
        else:
            print(f"Skipped '{textbook}' due to empty or invalid first chapter.")

    # Log the number of valid embeddings we have before clustering
    print(f"Total valid embeddings created: {len(embeddings)}")
    
    if len(embeddings) < 2:
        print("Not enough valid textbooks for clustering. At least 2 are required.")
        return {}

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    # Find optimal number of clusters using silhouette score
    best_n_clusters = find_optimal_clusters(embeddings, max_k=max_k)
    # Override
    best_n_clusters = 10

    # Perform Agglomerative Clustering with the optimal number of clusters
    clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
    labels = clustering.fit_predict(embeddings)

    # Group textbooks by their cluster labels
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(list(first_chapters.keys())[idx])

    return clusters

def main():
    textbook_path = 'D:/Textbooks'
    textbook_groups = organize_textbooks_by_similarity(textbook_path, max_k=10)
    
    if textbook_groups:
        for cluster, group in textbook_groups.items():
            print(f"Cluster {cluster}:")
            for textbook in group:
                print(f"  {textbook}")
    else:
        print("No valid textbook groups were created.")

if __name__ == '__main__':
    main()
