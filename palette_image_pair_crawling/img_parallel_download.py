import concurrent.futures
import io
import urllib.request
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_folder_path = r"E:\palette_and_images\Image\\"
color_folder_path = r"E:\palette_and_images\Palette\\"


def color_numpy(rgb):
    total_width, total_height = 512, 512

    repeat_count = total_width // len(rgb)

    color_image = np.repeat(rgb, repeat_count, axis=0)

    remaining_rows = total_width % len(rgb)
    if remaining_rows > 0:
        remaining_colors = np.repeat([rgb[-1]], remaining_rows, axis=0)
        color_image = np.concatenate([color_image, remaining_colors], axis=0)

    color_image = np.tile(color_image[np.newaxis, ...], (total_height, 1, 1))

    return color_image.astype(np.uint8)


def crop_to_square(image):
    height, width = image.shape[:2]

    min_dim = min(height, width)

    start_h = (height - min_dim) // 2
    end_h = start_h + min_dim
    start_w = (width - min_dim) // 2
    end_w = start_w + min_dim

    cropped_image = image[start_h:end_h, start_w:end_w]

    return cropped_image

img_counter = 0

def download_image(url, color, img_number):
    global img_counter

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        img_url = urllib.request.urlopen(request).read()
        image = Image.open(io.BytesIO(img_url))

        img_array = np.array(image)
        image_array = crop_to_square(img_array)
        color_image = color_numpy(color)

        image_path = image_folder_path + f"{img_number}.png"
        color_path = color_folder_path + f"{img_number}.png"
        
        Image.fromarray(image_array).save(image_path, format='PNG')
        Image.fromarray(color_image).save(color_path, format='PNG')

        img_counter += 1
        if img_counter%1000 == 0:
            print(f"{img_counter} images saved")
    except Exception as e:
        print(f"Error downloading {img_number} {url}: {e}")


def main():
    with open("palette_and_image.jsonl", "r") as file:
        data_list = [json.loads(line) for line in file]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        download_tasks = [
            (data["src"], data["rgb_values"], idx) for idx, data in enumerate(data_list)
        ]
        print(f"{len(download_tasks)} images")
        executor.map(lambda args: download_image(*args), download_tasks)


if __name__ == "__main__":
    main()
