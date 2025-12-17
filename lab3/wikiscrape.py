import time
from asyncio import wait

import requests
import re
from bs4 import BeautifulSoup

en_rand_article = "https://en.wikipedia.org/wiki/Special:Random"
nl_rand_article = "https://nl.wikipedia.org/wiki/Speciaal:Willekeurig"
headers = {'User-Agent': 'EnBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}

def scrape():
    for i in range(300):
        response = requests.get(en_rand_article, headers=headers)
        r = re.compile("<p>([\S\s]*?)</p>")
        data = r.findall(response.text)
        cleaned = [BeautifulSoup(item, "html.parser").get_text() for item in data]
        if len(cleaned) >= 1:
            for j in range(1):
                cleaned_seg = cleaned[j]
                clean_len = len(cleaned_seg.split())
                if clean_len >= 15:
                    first_15 = " ".join(cleaned_seg.split()[:15])
                    ex = "en|" + first_15
                    print(ex)
            time.sleep(3)
        response = requests.get(nl_rand_article, headers=headers)
        r = re.compile("<p>([\S\s]*?)</p>")
        data = r.findall(response.text)
        cleaned = [BeautifulSoup(item, "html.parser").get_text() for item in data]
        if len(cleaned) >= 2:
            for j in range(2):
                cleaned_seg = cleaned[j]
                clean_len = len(cleaned_seg.split())
                if clean_len >= 15:
                    first_15 = " ".join(cleaned_seg.split()[:15])
                    ex = "nl|" + first_15
                    print(ex)
            time.sleep(3)


def main():
    scrape()

if __name__ == '__main__':
    main()