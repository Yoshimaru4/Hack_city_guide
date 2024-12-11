import os
import csv
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://um.mos.ru"
CATEGORIES = ["houses", "places", "monuments"]

def get_max_pages(category):
    """
    Fetch the first page for a given category and determine the maximum number of pages
    by looking at the pagination links.
    """
    url = f"{BASE_URL}/{category}/?page=1"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the first page for {category}, status code: {response.status_code}")
        return 1  # default to 1 if something fails
    
    soup = BeautifulSoup(response.text, "html.parser")
    # Find all pagination links
    pagination_links = soup.find_all("a", class_="Pagination_paginationList__link__mp7bO")
    if not pagination_links:
        # If no pagination links found, assume only one page
        return 1
    
    # Extract the maximum page number from the pagination links
    page_numbers = []
    for link in pagination_links:
        text = link.get_text(strip=True)
        if text.isdigit():
            page_numbers.append(int(text))
    
    return max(page_numbers) if page_numbers else 1

def scrape_category(category):
    max_pages = get_max_pages(category)

    # Ensure datasets folder exists
    os.makedirs("datasets", exist_ok=True)
    output_file = os.path.join("datasets", f"{category}.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(["Name", "Link"])
        
        for page_num in tqdm(range(1, max_pages + 1), desc=f"Scraping {category}", unit="page"):
            url = f"{BASE_URL}/{category}/?page={page_num}"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Warning: Page {page_num} of {category} returned status {response.status_code}, skipping.")
                continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            # Find all relevant links
            anchors = soup.find_all("a", class_="AbstractCard_title__Z2Hu2")
            if not anchors:
                # Might be no records on this page
                continue
            
            for a in anchors:
                name = a.get_text(strip=True)
                relative_link = a.get("href", "")
                if relative_link.startswith("/"):
                    full_link = BASE_URL + relative_link
                else:
                    full_link = relative_link
                writer.writerow([name, full_link])
    
    print(f"Data scraping completed for {category}. Check {output_file} for results.")

if __name__ == "__main__":
    for cat in CATEGORIES:
        scrape_category(cat)
