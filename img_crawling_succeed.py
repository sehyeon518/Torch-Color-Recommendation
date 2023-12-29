from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import requests

PAUSE_TIME = 1
query = "product"

options = webdriver.ChromeOptions()
# options.add_argument('headless')
# options.add_argument('--disable-gpu')
# options.add_argument('lang=ko_KR')
# options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36")

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
driver.get('https://pixabay.com/images/search/'+query)

img_elements = driver.find_elements(By.TAG_NAME, 'img')

print(len(img_elements))

imgs = []

for i, img in enumerate(img_elements):
    img_url = (img.get_attribute('src'))
    save_name = 'Images\product_images\\' + 'product_' + str(i) + '.jpg'
    response = requests.get(img_url)
    try:
        if response.status_code == 200:
            with open(save_name, 'wb') as file:
                file.write(response.content)
    except:
        pass

driver.quit()
