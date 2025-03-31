import asyncio
import os
import logging
import colorlog
from rnet import Impersonate, Client
from bs4 import BeautifulSoup
import random
import json
import string
from time import sleep
from datetime import datetime

# Configure logging
logger = logging.getLogger("scraper")
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

RESUME_FILE = "resume.txt"
BASE_URL = "https://www.kleinanzeigen.de"

# Define the folder path
# Ensure the directory exists
folder = os.path.join(os.getcwd(), "extract_data")
os.makedirs(folder, exist_ok=True)



async def get_html(client, url):
    """Fetch the page content asynchronously."""
    try:
        logger.info(f"Fetching: {url}")
        resp = await client.get(url)
        return await resp.text()
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def save_resume_point(url):
    """Save the last successful page URL to resume later."""
    with open(RESUME_FILE, "w") as f:
        f.write(url)

def load_resume_point():
    """Load the last saved page URL."""
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "r") as f:
            url = f.read().strip()
            logger.info(f'RESUME POINT: {url}')
            return url
    return None

def extract_ads(html):
    """Extract ad details from the page."""
    soup = BeautifulSoup(html, "lxml")
    ads = []
    

    for li in soup.select(".aditem"):
        ad = {}
        try: 
            image_tag = li.select_one("div.aditem-image img")
            aditem_tag = li.select_one("div.aditem-main")
            bottom_tags = li.select('.aditem-main--bottom span')

            ad['id'] = li['data-adid']

            # each ad has bottom tags, when 3 tags the first is Gesuch, when 2 the first one is KM.
            match len(bottom_tags):
                case 3:
                    ad['ad_type'] = 'wanted' 
                    ad['car_km'] = bottom_tags[1].get_text(strip=True).strip('km').replace(".", "")
                    ad['car_registration'] = bottom_tags[2].get_text(strip=True).strip('EZ')
                case 2:
                    ad['ad_type'] = 'offer' 
                    ad['car_km'] = bottom_tags[0].get_text(strip=True).strip('km').replace(".", "")
                    ad['car_registration'] = bottom_tags[1].get_text(strip=True).strip('EZ')
                case _:
                    continue

            if image_tag:
                ad['image_src'] = image_tag["src"]
            
            if aditem_tag:
                ad['location'] = aditem_tag.select_one(".aditem-main--top--left").get_text(strip=True)
                ad['date_posted'] = aditem_tag.select_one(".aditem-main--top--right").get_text(strip=True)
                ad['date_scrape'] = datetime.now().strftime("%m/%d/%Y")
                ad['title'] = aditem_tag.select_one(".text-module-begin").get_text(strip=True)
                ad['price'] = aditem_tag.select_one(".aditem-main--middle--price-shipping--price").get_text(strip=True).strip("€").replace(".", "")
            
            ads.append(ad)    
        except Exception as e:
            logger.error(f"Error processing ad: {e}", exc_info=True)

    return ads

def get_next_page(html):
    """Find the next page URL."""
    soup = BeautifulSoup(html, "lxml")
    next_page_tag = soup.select_one(".pagination-next")
    
    href = None
    data_url = None
    next_url = None

    if next_page_tag:
        try:
            href = next_page_tag["href"]
        except:
            logger.info('paginatination href not found')
        
        try:
            data_url= next_page_tag["data-url"]
        except:
            logger.info('paginatination data-url not found')

        next_url = href or data_url  

    if next_url:
        logger.info(f'NEXT PAGE: {next_url}')
        return BASE_URL + next_url
    return None

def write_to_file(ads):
    try:
        # Generate a random filename
        filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".json"
        filepath = os.path.join(folder, filename)

        # Write dictionary to JSON file (overwrite mode)
        with open(filepath, "w") as file:
            json.dump(ads, file, indent=4)

        logger.info(f"Data written to {filepath}")
        return True
    except:
        logger.error(f"Did not write content: {ads}")
        return False

async def scrape_ads(start_url):
    """Main scraping function with pagination and error handling."""
    client = Client(impersonate=Impersonate.Firefox136)
    url = load_resume_point() or start_url  # Resume from last saved point

    while url:
        sleeptime = random.randint(2,5)
        logger.info(f"sleeping for:{sleeptime} seconds")
        sleep(sleeptime)

        html = await get_html(client, url)
        if not html:
            logger.error("Skipping page due to fetch error.")
            continue
        else:
            ads = extract_ads(html)
            if ads:
                write_to_file(ads)

                #DEBUG
                if logger.isEnabledFor(logging.DEBUG):
                    filepath = filepath + ".html"
                    with open(filepath, "w", encoding="utf-8") as file:
                        file.write(html)
                        logger.debug(f"HTML written to {filepath}")


            save_resume_point(url)  # Save progress

            next_page = get_next_page(html)
            if next_page:
                url = next_page
                logger.info(f"Next page found: {url}")
            else:
                logger.info("No more pages to scrape.")
                break

    

async def main():
    """Start the scraping process."""
    start_url = "https://www.kleinanzeigen.de/s-autos/dacia/c216+autos.marke_s:dacia"
    await scrape_ads(start_url)

if __name__ == "__main__":
    asyncio.run(main())
