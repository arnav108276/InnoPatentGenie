
from flask import Flask, request, jsonify, render_template
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import ollama
import pinecone
import logging
import time
import uuid

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Initialize Pinecone
pinecone.init(api_key="pcsk_3L6FMG_Q9feJUoC1TrF3dspXkuzyFVTLWR9kEPdpnTsXtko4yH3XdwkCWHZw4xA5Lan9m6", environment="YOUR_ENV")
index = pinecone.Index("your-index-name")

# Embedding function using Ollama
def get_ollama_embedding(text):
    response = ollama.embeddings(model="llama3-chatqa:8b", prompt=text)
    return response['embedding']

# Store scraped patent data into Pinecone
def store_in_pinecone(patents):
    for patent in patents:
        embedding = get_ollama_embedding(patent['title'] + " " + patent['description'])
        metadata = {
            "title": patent['title'],
            "description": patent['description'],
            "url": patent['url'],
            "id": patent['id'],
            "image": patent.get('image', ''),
            "filing_date": patent.get('filing_date', '')
        }
        index.upsert(vectors=[(str(uuid.uuid4()), embedding, metadata)])

# Retrieve top-k relevant patents from Pinecone
def retrieve_from_pinecone(query, top_k=5):
    query_embedding = get_ollama_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata'] for match in results['matches']]

# Web Scraping Functions (Google Patents & Espacenet)
def get_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def scrape_google_patents(query, page_num):
    results = []
    driver = get_chrome_driver()
    try:
        url = f"https://patents.google.com/?q={query}&oq={query}&page={page_num}"
        driver.get(url)
        time.sleep(2) 
        items = driver.find_elements(By.CSS_SELECTOR, 'search-result-item')
        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, '#htmlContent').text.strip()
                description = item.find_element(By.CSS_SELECTOR, '#htmlContent').text.strip()
                patent_id = item.find_element(By.CSS_SELECTOR, '[data-proto="OPEN_PATENT_PDF"]').text.strip()
                patent_url = f"https://patents.google.com/patent/{patent_id}/en"
                results.append({"title": title, "description": description, "id": patent_id, "url": patent_url})
            except Exception as e:
                print(f"Error extracting patent info: {e}")
    finally:
        driver.quit()
    return results

def scrape_espacenet(query, page_num):
    results = []
    driver = get_chrome_driver()
    try:
        url = f"https://worldwide.espacenet.com/patent/search?q={query}&page={page_num}"
        driver.get(url)
        time.sleep(2)
        items = driver.find_elements(By.CSS_SELECTOR, 'article.item--wSceB4di')
        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, 'header.h2--2VrrSjFb').text.strip()
                description = item.find_element(By.CSS_SELECTOR, '.copy-text--uk738M73').text.strip()
                patent_id = item.find_element(By.TAG_NAME, 'a').get_attribute('href').split('/')[-1]
                patent_url = f"https://patents.google.com/patent/{patent_id}/en"
                results.append({"title": title, "description": description, "id": patent_id, "url": patent_url})
            except Exception as e:
                print(f"Error extracting patent info: {e}")
    finally:
        driver.quit()
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    page = request.args.get('page', 1, type=int)
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    try:
        # Fetch new patents & store them in Pinecone
        espacenet_results = scrape_espacenet(query, page)
        google_results = scrape_google_patents(query, page)
        combined_results = google_results + espacenet_results
        store_in_pinecone(combined_results)
        
        # Retrieve top 5 relevant patents from Pinecone
        optimized_results = retrieve_from_pinecone(query, top_k=5)
        
        return jsonify({'results': optimized_results, 'page': page})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5044)
