import asyncio
from rnet import Impersonate , Client
from bs4 import BeautifulSoup
import logging
import colorlog
from datetime import datetime
import os
import random , string , json


formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Define the folder path
# Ensure the directory exists
folder = os.path.join(os.getcwd(), "extract_data_frontpage")
os.makedirs(folder, exist_ok=True)

def ex_kv(text):
    text= text.strip()
    parts = text.rsplit("(", 1)  # Split at the last opening bracket
    txt = parts[0].strip()  # Extract text part
    number = parts[1].rstrip(")") if len(parts) > 1 else None  # Extract number part
    # Remove the thousands separator (dot)
    if number:
        number = number.replace(".", "")
    return txt,number

def scrape_subject(html,subject):
    soup = BeautifulSoup(html, "html.parser")
    match subject:
        case "brand":
            elements = soup.select('ul[data-overlayheadline="Dacia"] > li')
        case "fuel":
            elements = soup.select('ul[data-overlayheadline="Kraftstoffart"] > li')
        case "euro":
            elements = soup.select('ul[data-overlayheadline="Schadstoffklasse"] > li')
        case "location":
            elements = soup.select('ul[data-overlayheadline="Ort"] > li')
        case _:
            elements = None
            logger.error(f'scrape_subject : invalid subject: {subject}')

    if elements is not None:
        i = 0
        categories = []
        for el in elements:
            i+=1
            category = {}
            a_tag = el.find("a", class_="text-link-subdued")
            if a_tag:
                link_href = a_tag["href"]  # Get href value

            txt = el.get_text()
            if txt:
                k,v = ex_kv(txt)
                category['date_scrape'] = datetime.now().strftime("%m/%d/%Y") 
                category['category'] = k
                category['count'] = v
                category['url'] = link_href
            else:
                logger.error(f'when scraping {subject}, in iter[{i}] k v could not be scrapped or found')
            
            categories.append(category)

        return categories

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

async def main():
    client = Client(impersonate=Impersonate.Firefox136)
    resp = await client.get("https://www.kleinanzeigen.de/s-autos/dacia/c216+autos.marke_s:dacia")
    html_content = await resp.text()

    

    write_to_file(scrape_subject(html_content,'brand'))
    write_to_file(scrape_subject(html_content,'fuel'))
    write_to_file(scrape_subject(html_content,'euro'))
    write_to_file(scrape_subject(html_content,'location'))
    
    
    
     

if __name__ == "__main__":
    asyncio.run(main())