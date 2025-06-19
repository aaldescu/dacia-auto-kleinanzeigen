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

RESUME_FILE = "resume_detailed.json"
BASE_URL = "https://www.kleinanzeigen.de"
DETAILED_AD_URLS_FILE = "urls_detailed_ad.txt"

# Ensure EXTRACT_DATA folder exists
folder = os.path.join(os.getcwd(), "extract_data_detailed")
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

def extract_detailed_ad(html):
    """Extract detailed information from an ad page."""
    soup = BeautifulSoup(html, "lxml")
    ad_details = {}
    
    try:
        # Extract user profile information
        user_profile = soup.select_one(".userprofile-vip")
        if user_profile:
            ad_details['seller_name'] = user_profile.get_text(strip=True)
            
        # Extract user type
        user_type = soup.select_one(".userprofile-vip-details-text")
        if user_type:
            ad_details['seller_type'] = user_type.get_text(strip=True)
            
        # Extract active since
        active_since_elements = soup.select(".userprofile-vip-details-text")
        if len(active_since_elements) > 1:
            ad_details['active_since'] = active_since_elements[1].get_text(strip=True)
            
        # Extract ad details
        details_container = soup.select_one("#viewad-details")
        if details_container:
            details_list = details_container.select(".addetailslist--detail")
            for detail in details_list:
                name = detail.get_text(strip=True).split('\n')[0].strip()
                value = detail.select_one(".addetailslist--detail--value").get_text(strip=True)
                if name and value:
                    #remove value from name
                    name = name.replace(value, "").strip()
                ad_details[name] = value
                
        # Extract extras/features
        extras_container = soup.select_one("#viewad-configuration")
        if extras_container:
            extras = extras_container.select(".checktag")
            if extras:
                ad_details['extras'] = [extra.get_text(strip=True) for extra in extras]
                
        # Extract description
        description = soup.select_one("#viewad-description-text")
        if description:
            ad_details['description'] = description.get_text(strip=True)
            
        # Extract seller contact info
        seller_info = soup.select_one("#viewad-contact")
        if seller_info:
            seller_name = seller_info.select_one(".userprofile-vip")
            if seller_name:
                ad_details['seller_name'] = seller_name.get_text(strip=True)
                
        # Extract title and price
        title = soup.select_one("h1#viewad-title")
        if title:
            ad_details['title'] = title.get_text(strip=True)
            
        price = soup.select_one("#viewad-price")
        if price:
            ad_details['price'] = price.get_text(strip=True)
            
            
        # Extract ad ID
        ad_id_box = soup.select_one("#viewad-ad-id-box")
        if ad_id_box:
            ad_id = ad_id_box.select_one("ul li:nth-child(2)")
            if ad_id:
                ad_details['ad_id'] = ad_id.get_text(strip=True)
    except Exception as e:
        logger.error(f"Error extracting detailed ad: {e}")
        
    return ad_details


def write_to_file(ads, detailed=False):
    """Write ads to a JSON file."""
    if not ads:
        logger.warning("No ads to write to file")
        return
    
    folder_name = "extract_data_detailed" if detailed else "extract_data"
    folder = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder, exist_ok=True)
    
    # Check if ad_id exists in the ad data
    if isinstance(ads, dict) and 'ad_id' in ads and ads['ad_id']:
        # Use ad_id for the filename
        filename = f"{ads['ad_id']}.json"
        logger.info(f"Using ad ID for filename: {filename}")
    else:
        # Use the current random name method
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        filename = f"ads_{timestamp}_{random_str}.json" if not detailed else f"detailed_ad_{timestamp}_{random_str}.json"
    
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ads, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Wrote {len(ads) if isinstance(ads, list) else 1} ad(s) to {filepath}")

async def scrape_detailed_ad(start_url, start_url_index=None):
    """Main scraping function with pagination and error handling."""
    current_url = start_url
    page_count = 1
    total_ads = 0
    
    # Create a client with impersonation
    client = Client(impersonate=Impersonate.Firefox136)
    while current_url:
        # Save resume point before fetching
        save_resume_point(current_url, start_url_index)
        
        # Fetch the page
        html = await get_html(client, current_url)
        if not html:
            logger.error(f"Failed to fetch {current_url}, skipping...")
            break
        
        # Extract detailed ad
        ad = extract_detailed_ad(html)
        if ad:
            total_ads += 1
            write_to_file(ad, detailed=True)
            
        # Add a small delay between requests to avoid rate limiting
        sleep(random.uniform(1, 3))
        
        # We've processed this URL, so set current_url to None to exit the loop
        # This is needed because we're directly processing specific ad URLs, not paginating
        current_url = None
        
        sleep(random.uniform(2, 5))
            
    logger.info(f"Completed scraping {page_count} pages with {total_ads} ads")
    return total_ads

async def main():
    """Start the scraping process."""
    logger.info("Starting scraper...")
    
    # start URLs 
    # start_urls variable is populated from file urls_detailed_ad.txt 
    start_urls = []
    with open(DETAILED_AD_URLS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            start_urls.append(line.strip())

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
                    await scrape_detailed_ad(resume_url, current_index)
                    # Move to the next start URL
                    current_index += 1
                    break
            else:
                logger.warning("Could not determine which start URL the resume point belongs to. Starting from the beginning.")

    # Process remaining start URLs
    for i in range(current_index, len(start_urls)):
        start_url = start_urls[i]
        logger.info(f"Starting to scrape URL {i+1}/{len(start_urls)}: {start_url}")
        await scrape_detailed_ad(start_url, i)
    
    # All start URLs have been processed
    logger.info("✅ All start URLs have been processed. Scraping complete!")
    
    # Reset resume point to start from the beginning next time
    if os.path.exists(RESUME_FILE):
        os.remove(RESUME_FILE)
        logger.info("Resume point has been reset for the next run.")
    
    # Delete the DETAILED_AD_URLS_FILE
    if os.path.exists(DETAILED_AD_URLS_FILE):
        os.remove(DETAILED_AD_URLS_FILE)
        logger.info("Deleted DETAILED_AD_URLS_FILE")

if __name__ == "__main__":
    asyncio.run(main())
