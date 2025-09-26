from bs4 import BeautifulSoup
import requests
import urllib3
from fake_useragent import UserAgent
import json
import csv
import time

# Disable SSL warnings when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ua = UserAgent()

# Headers to mimic a real browser
headers = {
    "User-Agent": ua.random,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "X-Requested-With": "XMLHttpRequest",  # Important for AJAX requests
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}


def fetch_page_data(page_start, page_length=25):
    """
    Fetch data from a specific page using the AJAX endpoint

    Args:
        page_start: Starting record number (0-based)
        page_length: Number of records per page
    """

    # AJAX endpoint URL
    ajax_url = "https://www.d3pharma.com/D3DistalMutation/loadIncreaseData.php"

    # DataTables sends these parameters for server-side processing
    data = {
        "draw": 1,  # Draw counter
        "start": page_start,  # Starting record
        "length": page_length,  # Number of records per page
        "search[value]": "",  # Search value
        "search[regex]": "false",
        "order[0][column]": "0",  # Order by first column
        "order[0][dir]": "asc",  # Ascending order
    }

    # Add column definitions (based on the table structure we saw)
    columns = [
        "enzyme_name",
        "abbreviation",
        "uniprot_id",
        "pdb_id",
        "pdb_chain",
        "mutation_resi",
        "mutation_type",
        "active_site_resi",
        "active_resn",
        "distance",
        "activity_change",
        "references",
    ]

    for i, col in enumerate(columns):
        data[f"columns[{i}][data]"] = col
        data[f"columns[{i}][name]"] = ""
        data[f"columns[{i}][searchable]"] = "true"
        data[f"columns[{i}][orderable]"] = "true" if i not in [4, 6, 8, 11] else "false"
        data[f"columns[{i}][search][value]"] = ""
        data[f"columns[{i}][search][regex]"] = "false"

    try:
        # First try with SSL verification
        response = requests.post(ajax_url, headers=headers, data=data, timeout=10)
    except requests.exceptions.SSLError:
        print("SSL verification failed, trying without verification...")
        # Fallback to no SSL verification
        response = requests.post(
            ajax_url, headers=headers, data=data, verify=False, timeout=10
        )
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if response.status_code == 200:
        try:
            json_data = response.json()
            return json_data
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response for page starting at {page_start}")
            print("Response content:", response.text[:500])
            return None
    else:
        print(f"HTTP Error {response.status_code} for page starting at {page_start}")
        return None


def fetch_all_data():
    """
    Fetch data from all 80 pages
    """
    all_data = []
    page_length = 25
    total_records = 1985  # From the HTML: "deferLoading":1985
    total_pages = (total_records + page_length - 1) // page_length  # Ceiling division

    print(f"Fetching data from {total_pages} pages ({total_records} total records)...")

    for page in range(total_pages):
        page_start = page * page_length
        print(
            f"Fetching page {page + 1}/{total_pages} (records {page_start}-{page_start + page_length - 1})..."
        )

        page_data = fetch_page_data(page_start, page_length)

        if page_data and "data" in page_data:
            all_data.extend(page_data["data"])
            print(f"  -> Got {len(page_data['data'])} records")

            # Add a small delay to be respectful to the server
            time.sleep(0.5)
        else:
            print(f"  -> Failed to get data for page {page + 1}")

    return all_data


def save_to_csv(data, filename="enzyme_mutations.csv"):
    """
    Save data to CSV file
    """
    if not data:
        print("No data to save")
        return

    # Column headers
    headers = [
        "enzyme_name",
        "abbreviation",
        "uniprot_id",
        "pdb_id",
        "pdb_chain",
        "mutation_resi",
        "mutation_type",
        "active_site_resi",
        "active_resn",
        "distance",
        "activity_change",
        "references",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for row in data:
            # Clean HTML tags from the data
            cleaned_row = []
            for cell in row:
                if isinstance(cell, str):
                    # Remove HTML tags using BeautifulSoup
                    soup = BeautifulSoup(cell, "html.parser")
                    cleaned_cell = soup.get_text(strip=True)
                    cleaned_row.append(cleaned_cell)
                else:
                    cleaned_row.append(cell)
            writer.writerow(cleaned_row)

    print(f"Data saved to {filename}")


if __name__ == "__main__":
    print("Starting to fetch all enzyme mutation data...")

    # Fetch all data
    all_data = fetch_all_data()

    if all_data:
        print(f"\nSuccessfully fetched {len(all_data)} total records")

        # Save to CSV
        save_to_csv(all_data, "enzyme_mutations_all_pages.csv")

        # Save to JSON as backup
        with open("enzyme_mutations_all_pages.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print("Data also saved to enzyme_mutations_all_pages.json")

    else:
        print("Failed to fetch data")
