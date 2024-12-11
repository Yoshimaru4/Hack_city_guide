import os
import csv
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from time import sleep
BASE_URL = "https://um.mos.ru"
CATEGORIES = ["houses", "places", "monuments"]

# We only apply this start index to the 'houses' category
start_index = 1  # Resume from this entry number for houses

def parse_detail_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None, None, None

    soup = BeautifulSoup(response.text, "html.parser")

    # Name
    name_div = soup.find("div", class_="ItemDetailPageLayout_card__title__VBGt1")
    name = name_div.get_text(strip=True) if name_div else ""

    # Geo location
    geo_span = soup.find("span", class_="InfoTags_groupItems__label__XycSo")
    geo = geo_span.get_text(strip=True) if geo_span else ""

    # Info text: gather paragraphs from all `div.CardBox_article__1tzZc`
    info_text = []
    info_boxes = soup.find_all("div", class_="CardBox_article__1tzZc")
    for box in info_boxes:
        paragraphs = box.find_all("p")
        for p in paragraphs:
            info_text.append(p.get_text(strip=True))
    full_info = "\n".join(info_text)

    return name, geo, full_info

def main():
    os.makedirs("datasets", exist_ok=True)
    output_file = os.path.join("datasets", "details.csv")

    # Overwrite details.csv. If you need to append instead, change "w" to "a" and handle headers accordingly.
    with open(output_file, "w+", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(["Category", "Name", "Geo", "Info", "SourceLink"])

        for category in CATEGORIES:
            input_file = os.path.join("datasets", f"{category}.csv")
            if not os.path.exists(input_file):
                print(f"Input file {input_file} not found. Skipping {category}.")
                continue

            # Read entries from the category file
            entries = []
            with open(input_file, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) == 2:
                        name, link = row
                        entries.append((name, link))

            # Determine slicing logic based on category
            if category == "asd":
                # Resume from start_index for houses
                total_entries = len(entries)
                if start_index > total_entries:
                    print(f"Start index {start_index} exceeds total entries {total_entries} for houses.")
                    sliced_entries = []
                else:
                    sliced_entries = entries[start_index-1:]
                desc_str = f"Parsing {category} from {start_index}"
            else:
                # Start from the beginning for other categories
                sliced_entries = entries
                desc_str = f"Parsing {category} from start"

            for (name, link) in tqdm(sliced_entries, desc=desc_str, unit="item"):
                try:
                    detail_name, geo, info_text = parse_detail_page(link)
                    if not detail_name:
                        detail_name = name
                    writer.writerow([category, detail_name, geo, info_text, link])
                    sleep(1)
                except:
                    pass

    print(f"All details combined into {output_file}.")

if __name__ == "__main__":
    main()
