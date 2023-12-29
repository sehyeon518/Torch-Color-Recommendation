from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

options = webdriver.ChromeOptions()

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

time.sleep(3)

count = 0
with open('image_urls.txt', 'a') as file:  # Open a file in append mode

    for i in range(355):
        driver.get(fr'https://pixabay.com/images/search/digital%20art/?pagi={i+1}')
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        imgs = soup.select('div.container--MwyXl img')

        for img in imgs:
            srcset = img.get('data-lazy-srcset')
            if srcset is None:
                srcset = img.get('srcset')

            if srcset is None:
                continue

            src = ""
            if len(srcset):
                src = str(srcset).split()[0]
                file.write(src + '\n')  # Append the src to the file
                count += 1
                
        print(f'{i} page, {count}')

    driver.quit()
