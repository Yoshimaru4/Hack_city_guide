import os
import csv
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

BASE_URL = "https://um.mos.ru"
CATEGORIES = ["houses", "places", "monuments"]
PAGE_LOAD_DELAY = 3  # seconds to wait for page to load

def initialize_driver():
    """
    Initialize the Selenium WebDriver with desired options.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode for efficiency
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    return driver

def get_max_pages(driver, category):
    """
    Fetch the first page for a given category and determine the maximum number of pages
    by looking at the pagination links.
    """
    url = f"{BASE_URL}/{category}/?page=1"
    driver.get(url)
    time.sleep(PAGE_LOAD_DELAY)  # Wait for the page to load

    soup = BeautifulSoup(driver.page_source, "html.parser")
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

def scrape_category(driver, category):
    max_pages = get_max_pages(driver, category)

    # Ensure datasets folder exists
    os.makedirs("datasets", exist_ok=True)
    output_file = os.path.join("datasets", f"{category}.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(["Name", "Link"])
        
        for page_num in tqdm(range(1, max_pages + 1), desc=f"Scraping {category}", unit="page"):
            url = f"{BASE_URL}/{category}/?page={page_num}"
            driver.get(url)
            time.sleep(PAGE_LOAD_DELAY)  # Wait for the page to load
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
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

def main():
    driver = initialize_driver()
    try:
        for cat in CATEGORIES:
            scrape_category(driver, cat)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
