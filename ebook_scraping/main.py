import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

import pandas as pd
import requests
import chromedriver_autoinstaller
from config import google_api_key, google_cse_id

# Streamlit UI
st.title("Ebook Finder")

# Sidebar for category and subject selection
categories = {
    "Engineering": ["Chemical Engineering", "Civil Engineering", "Mechanical Engineering"],
    "Medical Science": ["Anatomy", "Cardiology", "Pharmacology"]
}

category = st.sidebar.selectbox("Select Category", list(categories.keys()))
subject = st.sidebar.selectbox("Select Subject", categories[category])



# Function to get total hits using Google Custom Search API
def get_total_hits(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={google_cse_id}&num=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        total_results = data.get('queries', {}).get('request', [{}])[0].get('totalResults', "0")
        return int(total_results)
    else:
        st.error(f"Failed to fetch total results. Error: {response.text}")
        return 0

# Function to search for PDFs using Selenium
def search_pdfs(subject):
    query = f"{subject} filetype:pdf"
    search_url = f"https://www.google.com/search?q={query}"
    driver = None
    chromedriver_autoinstaller.install()
    chrome_options = webdriver.ChromeOptions()

    # Configure Selenium WebDriver
    # chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                              options=chrome_options)

    # Open Google Search
    driver.get(search_url)

    # Extract PDF links
    pdf_links = []
    try:
        search_results = driver.find_elements(By.CSS_SELECTOR, 'a[jsname="UWckNb"]')
        for element in search_results:
            href = element.get_attribute('href')
            if href and 'pdf' in href:
                title_element = element.find_element(By.TAG_NAME, 'h3')  # Locate title inside the anchor
                title = title_element.text if title_element else "No title available"
                pdf_links.append({
                    "title": title,
                    "link": href,
                })
    except Exception as e:
        print(f"Error while extracting links: {e}")
    finally:
        driver.quit()
    return pdf_links

# Button to trigger the search
if st.button("Search for PDFs"):
    st.write(f"Searching for PDFs related to {subject}...")

    # Get total hits using Google Search API
    total_hits = get_total_hits(f"{subject} filetype:pdf")
    st.write(f"Total hits for {subject}: {total_hits}")

    # Search PDFs using Selenium
    pdf_links = search_pdfs(subject)

    if pdf_links:
        # Convert to DataFrame
        df = pd.DataFrame(pdf_links)
        df['total_hits'] = total_hits  # Add total hits as a new column

        st.subheader("Found PDFs:")
        st.dataframe(df)
    else:
        st.write("No PDFs found.")