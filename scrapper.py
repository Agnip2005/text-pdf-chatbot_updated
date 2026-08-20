import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================
BASE_URL = "https://www.tnu.in"
OUTPUT_DIR = r"D:\desktop\new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Pages to scrape (add more as needed)
PAGES_TO_SCRAPE = [
    "/",
    "/about",
    "/about-us",
    "/academics",
    "/admissions",
    "/admission",
    "/courses",
    "/departments",
    "/faculty",
    "/research",
    "/campus-life",
    "/contact",
    "/contact-us",
    "/placements",
    "/events",
    "/news",
    "/gallery",
    "/facilities",
    "/scholarship",
    "/fee-structure",
    "/hostel",
    "/library",
]

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    """Clean and normalize scraped text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-@#%&/]', '', text)
    return text.strip()


def is_valid_url(url):
    """Check if URL belongs to the same domain."""
    parsed = urlparse(url)
    return parsed.netloc in ["www.tnu.in", "tnu.in", ""]


def get_all_links(soup, base_url):
    """Extract all internal links from a page."""
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        if is_valid_url(full_url) and full_url.startswith("http"):
            links.add(full_url)
    return links


# ============================================================
#  MAIN SCRAPER CLASS
# ============================================================

class TNUScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.visited_urls = set()
        self.all_data = []
        self.failed_urls = []

    # ----------------------------------------------------------
    def fetch_page(self, url):
        """Fetch a single page and return BeautifulSoup object."""
        try:
            print(f"  ⏳ Fetching: {url}")
            response = self.session.get(url, timeout=15, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml")
            return soup, response.status_code
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to fetch {url} → {e}")
            self.failed_urls.append(url)
            return None, None

    # ----------------------------------------------------------
    def extract_page_data(self, soup, url):
        """Extract all useful content from a page."""
        data = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "title": "",
            "meta_description": "",
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "tables": [],
            "links": [],
            "contact_info": {},
            "full_text": "",
        }

        # ── Title ──────────────────────────────────────────────
        title_tag = soup.find("title")
        if title_tag:
            data["title"] = clean_text(title_tag.get_text())

        # ── Meta Description ───────────────────────────────────
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            data["meta_description"] = clean_text(meta.get("content", ""))

        # ── Remove Unwanted Tags ───────────────────────────────
        for tag in soup(["script", "style", "noscript",
                         "header", "footer", "nav", "iframe"]):
            tag.decompose()

        # ── Headings ───────────────────────────────────────────
        for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            for tag in soup.find_all(level):
                text = clean_text(tag.get_text())
                if text:
                    data["headings"].append({
                        "level": level,
                        "text": text
                    })

        # ── Paragraphs ─────────────────────────────────────────
        for p in soup.find_all("p"):
            text = clean_text(p.get_text())
            if text and len(text) > 20:          # skip tiny snippets
                data["paragraphs"].append(text)

        # ── Lists ──────────────────────────────────────────────
        for ul in soup.find_all(["ul", "ol"]):
            items = []
            for li in ul.find_all("li"):
                text = clean_text(li.get_text())
                if text:
                    items.append(text)
            if items:
                data["lists"].append(items)

        # ── Tables ─────────────────────────────────────────────
        for table in soup.find_all("table"):
            table_data = []
            for row in table.find_all("tr"):
                row_data = []
                for cell in row.find_all(["td", "th"]):
                    row_data.append(clean_text(cell.get_text()))
                if row_data:
                    table_data.append(row_data)
            if table_data:
                data["tables"].append(table_data)

        # ── Links ──────────────────────────────────────────────
        for a in soup.find_all("a", href=True):
            text = clean_text(a.get_text())
            href = urljoin(url, a["href"])
            if text:
                data["links"].append({
                    "text": text,
                    "href": href
                })

        # ── Contact Info (regex) ───────────────────────────────
        page_text = soup.get_text()
        emails = re.findall(
            r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
            page_text
        )
        phones = re.findall(
            r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
            page_text
        )
        if emails:
            data["contact_info"]["emails"] = list(set(emails))
        if phones:
            data["contact_info"]["phones"] = list(set(phones))

        # ── Full Text ──────────────────────────────────────────
        full_text_parts = []
        if data["title"]:
            full_text_parts.append(f"Page Title: {data['title']}")
        for h in data["headings"]:
            full_text_parts.append(f"{h['level'].upper()}: {h['text']}")
        full_text_parts.extend(data["paragraphs"])
        for lst in data["lists"]:
            full_text_parts.extend(lst)
        data["full_text"] = "\n".join(full_text_parts)

        return data

    # ----------------------------------------------------------
    def crawl(self, max_pages=50):
        """
        Crawl TNU website starting from predefined pages
        then auto-discover new links.
        """
        print("\n" + "="*60)
        print("   TNU WEBSITE SCRAPER - Starting...")
        print("="*60 + "\n")

        # Seed URLs
        urls_to_visit = set()
        for path in PAGES_TO_SCRAPE:
            urls_to_visit.add(urljoin(BASE_URL, path))

        # BFS crawl
        while urls_to_visit and len(self.visited_urls) < max_pages:
            url = urls_to_visit.pop()

            if url in self.visited_urls:
                continue

            self.visited_urls.add(url)
            soup, status = self.fetch_page(url)

            if soup is None:
                continue

            # Extract data
            page_data = self.extract_page_data(soup, url)
            self.all_data.append(page_data)
            print(f"  ✅ Scraped [{len(self.visited_urls)}]: {page_data['title'][:50]}")

            # Discover new links
            new_links = get_all_links(soup, url)
            for link in new_links:
                if link not in self.visited_urls:
                    urls_to_visit.add(link)

            time.sleep(1)   # be polite – don't hammer the server

        print(f"\n✅ Crawling done! Total pages scraped: {len(self.all_data)}")
        return self.all_data

    # ----------------------------------------------------------
    def save_data(self):
        """Save scraped data in multiple formats for RAG."""

        # ── 1. Raw JSON (all structured data) ──────────────────
        json_path = os.path.join(OUTPUT_DIR, "tnu_raw_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.all_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON saved  → {json_path}")

        # ── 2. Plain Text (best for RAG chunking) ──────────────
        txt_path = os.path.join(OUTPUT_DIR, "tnu_full_text.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for page in self.all_data:
                f.write(f"\n{'='*60}\n")
                f.write(f"SOURCE URL : {page['url']}\n")
                f.write(f"PAGE TITLE : {page['title']}\n")
                f.write(f"SCRAPED AT : {page['timestamp']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(page["full_text"])
                f.write("\n\n")
        print(f"📄 TXT saved   → {txt_path}")

        


# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    scraper = TNUScraper()
    scraper.crawl(max_pages=80)     # increase for deeper crawl
    scraper.save_data()

    print("\n" + "="*60)
    print("  ALL DONE! Files are in  D:\\desktop\\new\\")
    print("="*60)