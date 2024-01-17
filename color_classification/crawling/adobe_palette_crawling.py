import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

time.sleep(3)

driver.get(fr'https://color.adobe.com/ko/explore')

time.sleep(10)


count = 0
page = 1
while count < 100000:
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    colors = soup.find_all('div', class_='Theme__theme___FNytP')
    
    for color in colors:
        rgb_list = []
        swatch = color.find_all('div', class_='Swatch__swatch___Y5UYS')
        for s in swatch:
            style_attribute = s['style']
            rgb_value = [int(x) for x in style_attribute.split('rgb(')[1].split(')')[0].split(',')]
            rgb_list.append(rgb_value)

        with open("/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/jsonl/five_colors_palette.jsonl", 'a') as file:
            data = {}
            data["palette"] = rgb_list
            json.dump(data, file)
            file.write('\n')
            count += 1
        
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    button = driver.find_element(By.XPATH, '//*[@id="color-root"]/div/div[5]/button[3]')
    button.click()

    print(f"page: {page}, count: {count}")
    page += 1

