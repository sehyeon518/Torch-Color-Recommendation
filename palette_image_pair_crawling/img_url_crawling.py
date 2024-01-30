import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

options = webdriver.ChromeOptions()

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
driver.get(fr'https://color.adobe.com/ko/explore')

time.sleep(10)

count = 0
page = 1

while count < 100000:
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    card_elements = soup.find_all('div', {'class': 'Card__card___rqCLW'})
    
    for card_element in card_elements:
        image_element = card_element.find('img')
        image_src = image_element.get('src')
        rgb_values = [
            [int(value) for value in re.findall(r'\d+', div['style'])]
            for div in card_element.find_all('div', {'class': 'Swatch__swatch___Y5UYS'})
        ]

        if image_src and rgb_values:
            image_info = {
                'src': image_src,
                'rgb_values': rgb_values
            }

            # Append each dictionary as a separate line in a JSON Lines file
            with open('palette_and_image.jsonl', 'a') as jsonl_file:
                json_line = json.dumps(image_info)
                jsonl_file.write(json_line + '\n')

            count += 1

    # Scroll down
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Click the button to load more content
    time.sleep(2)
    button = driver.find_element(By.XPATH, '//*[@id="color-root"]/div/div[5]/button[3]')
    button.click()

    time.sleep(2)  # Additional sleep to ensure stability

    page += 1
    print(f"page: {page}, count: {count}")


driver.quit()
