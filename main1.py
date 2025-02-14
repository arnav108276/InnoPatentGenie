from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import pinecone
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="YOUR_PINECONE_ENV")
index_name = "patent-search"

# Create index if not exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")
index = pinecone.Index(index_name)

# Function to scrape Google Patents
def scrape_google_patents(query):
    url = f"https://patents.google.com/?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    patents = []
    for result in soup.find_all("search-result-item", limit=5):
        title = result.find("h3").text.strip()
        link = "https://patents.google.com" + result.find("a")["href"]
        patents.append({"id": link, "title": title, "description": title, "source": "Google Patents"})
    return patents

# Function to scrape Espacenet
def scrape_espacenet(query):
    url = f"https://worldwide.espacenet.com/searchResults?query={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    patents = []
    for result in soup.find_all("div", class_="result", limit=5):
        title = result.find("span", class_="title").text.strip()
        link = "https://worldwide.espacenet.com" + result.find("a")["href"]
        patents.append({"id": link, "title": title, "description": title, "source": "Espacenet"})
    return patents

# Store patents in Pinecone
def store_patents_in_pinecone(patents):
    vectors = []
    for patent in patents:
        text = f"{patent['title']} {patent['description']}"
        vector = embedder.encode(text).tolist()
        vectors.append((patent['id'], vector, patent))  # Store metadata

    index.upsert(vectors)  # Insert into Pinecone

# Search similar patents in Pinecone
def search_pinecone(query):
    query_vector = embedder.encode(query).tolist()
    search_results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    patents = []
    for match in search_results['matches']:
        patents.append(match['metadata'])
    return patents

# Placeholder for GWO Optimization
def gwo_optimize_results(results, query):
    return sorted(results, key=lambda x: len(x['title']), reverse=True)  # Dummy ranking

# Search endpoint
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    pinecone_results = search_pinecone(query)
    
    if not pinecone_results:
        google_results = scrape_google_patents(query)
        espacenet_results = scrape_espacenet(query)
        combined_results = google_results + espacenet_results
        
        store_patents_in_pinecone(combined_results)
        
        optimized_results = gwo_optimize_results(combined_results, query)
    else:
        optimized_results = gwo_optimize_results(pinecone_results, query)

    return jsonify({'results': optimized_results})

if __name__ == '__main__':
    app.run(debug=True)
