import asyncio
import os
import logging
import colorlog
from logging.handlers import RotatingFileHandler
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

# File handler for errors only
error_log_file = os.path.join(os.getcwd(), "scraper_errors.log")
error_handler = RotatingFileHandler(
    error_log_file,
    maxBytes=5*1024*1024,  # 5MB max file size
    backupCount=5,         # Keep 5 backup files
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s\n'
))
logger.addHandler(error_handler)


# Set logging level overall
logger.setLevel(logging.INFO)

RESUME_FILE = "resume.json"
BASE_URL = "https://www.kleinanzeigen.de"


# Ensure EXTRACT_DATA folder exists
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

def save_resume_point(url, start_url_index=None):
    """Save the last successful page URL and start URL index to resume later."""
    resume_data = {
        "url": url,
        "start_url_index": start_url_index
    }
    with open(RESUME_FILE, "w") as f:
        json.dump(resume_data, f)

def load_resume_point():
    """Load the last saved page URL and start URL index."""
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, "r") as f:
                resume_data = json.load(f)
                url = resume_data.get("url")
                start_url_index = resume_data.get("start_url_index")
                logger.info(f'RESUME POINT: URL={url}, START_URL_INDEX={start_url_index}')
                return url, start_url_index
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading resume point: {e}")
            # Handle old format for backward compatibility
            try:
                with open(RESUME_FILE, "r") as f:
                    url = f.read().strip()
                    if url:
                        logger.info(f'RESUME POINT (old format): {url}')
                        return url, None
            except Exception:
                pass
    return None, None

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
            logger.error(
                f"Error processing ad {ad_id} : {e}",
                exc_info=True,
                extra={
                    'ad_id': ad['id'] or 'unknown'
                }
            )
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

async def scrape_ads(start_url, start_url_index=None):
    """Main scraping function with pagination and error handling."""
    client = Client(impersonate=Impersonate.Firefox136)
    # We don't load the resume point here anymore, as it's handled in main()
    url = start_url
    
    # Track if we've processed any pages for this start URL
    pages_processed = 0

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
                # Increment the pages processed counter
                pages_processed += 1

                #DEBUG
                if logger.isEnabledFor(logging.DEBUG):
                    filepath = os.path.join(folder, f"debug_{random.randint(1000, 9999)}.html")
                    with open(filepath, "w", encoding="utf-8") as file:
                        file.write(html)
                        logger.debug(f"HTML written to {filepath}")

            # Save progress with the current start_url_index
            save_resume_point(url, start_url_index)  

            next_page = get_next_page(html)
            if next_page:
                url = next_page
                logger.info(f"Next page found: {url}")
            else:
                logger.info(f"No more pages to scrape for this start URL. Processed {pages_processed} pages.")
                return  # Return to main to process the next start URL

    

async def main():
    """Start the scraping process."""
    #Dacia main url = "https://www.kleinanzeigen.de/s-autos/dacia/c216+autos.marke_s:dacia"
    start_urls = [
        "https://www.kleinanzeigen.de/s-autos/dacia/baden-wuerttemberg/c216l7970+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/bayern/c216l5510+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/berlin/c216l3331+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/brandenburg/c216l7711+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/bremen/c216l1+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/hamburg/c216l9409+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/hessen/c216l4279+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/mecklenburg-vorpommern/c216l61+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/niedersachsen/c216l2428+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/nordrhein-westfalen/c216l928+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/rheinland-pfalz/c216l4938+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/saarland/c216l285+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/sachsen/c216l3799+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/sachsen-anhalt/c216l2165+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/schleswig-holstein/c216l408+autos.marke_s:dacia",
        "https://www.kleinanzeigen.de/s-autos/dacia/thueringen/c216l3548+autos.marke_s:dacia"
    ]

    # Load resume point
    resume_url, start_url_index = load_resume_point()
    
    # Determine where to start
    current_index = 0
    if start_url_index is not None and 0 <= start_url_index < len(start_urls):
        # We have a valid start_url_index from the resume point
        current_index = start_url_index
        logger.info(f"Resuming from start URL index: {current_index}")
    elif resume_url:
        # Try to find the URL in the start_urls list
        for i, url in enumerate(start_urls):
            if url == resume_url:
                current_index = i
                logger.info(f"Found resume URL in start_urls at index {current_index}")
                break
        else:
            # If we're in the middle of paginating through a start URL
            logger.info(f"Resume URL is not a start URL, continuing from current pagination")
            # Find which start URL this resume point belongs to
            for i, start_url in enumerate(start_urls):
                # Check if the resume URL contains the base part of the start URL
                # This is a heuristic and might need adjustment based on URL patterns
                if start_url.split('?')[0] in resume_url.split('?')[0]:
                    current_index = i
                    logger.info(f"Resume URL appears to be from start URL at index {current_index}")
                    # Process the current page first
                    logger.info(f"Continuing pagination from: {resume_url}")
                    await scrape_ads(resume_url, current_index)
                    # Move to the next start URL
                    current_index += 1
                    break
            else:
                logger.warning("Could not determine which start URL the resume point belongs to. Starting from the beginning.")

    # Process remaining start URLs
    for i in range(current_index, len(start_urls)):
        start_url = start_urls[i]
        logger.info(f"Starting to scrape URL {i+1}/{len(start_urls)}: {start_url}")
        await scrape_ads(start_url, i)
    
    # All start URLs have been processed
    logger.info("✅ All start URLs have been processed. Scraping complete!")
    
    # Reset resume point to start from the beginning next time
    if os.path.exists(RESUME_FILE):
        os.remove(RESUME_FILE)
        logger.info("Resume point has been reset for the next run.")

if __name__ == "__main__":
    asyncio.run(main())
