import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

options = webdriver.ChromeOptions()

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

time.sleep(3)

count = 0

driver.get(fr'https://coolors.co/colors')
time.sleep(3)
button = driver.find_element(By.XPATH, '//*[@id="iubenda-cs-banner"]/div/div/div/div[3]/div[2]/button')
button.click()
time.sleep(0.5)

button = driver.find_element(By.XPATH, '//*[@id="modal-fabrizio"]/div/div[2]/div/a')
button.click()
time.sleep(0.5)


soup = BeautifulSoup(driver.page_source, 'html.parser')
colors = soup.find_all('div', class_='color-card')

body = driver.find_element(By.TAG_NAME, 'body')
for i in range(70):
    try:
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)
        print(i)
    except:
        pass

soup = BeautifulSoup(driver.page_source, 'html.parser')
colors = soup.find_all('div', class_='color-card')

for color in colors:
    color_div = color.find('div', class_='color-card_color')
    color_name = color.find('a', class_='color-card_name')

    if color_div and color_name:
        style_attribute = color_div.get('style')
        if style_attribute:
            background_value = style_attribute.split(":")[1].strip()[:-1]

            name_text = color_name.get_text(strip=True)

            color_data = {
                'color_name': name_text,
                'background_rgb': background_value
            }

            with open("/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/list_of_colors.jsonl", 'a') as file:
                json.dump(color_data, file)
                file.write('\n')
                count += 1

driver.quit()