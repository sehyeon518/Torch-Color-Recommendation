from PIL import Image
import os

def transparent(image_path):
    threshold = 0.5
    img = Image.open(image_path)
    alpha_channel = img.split()[-1]
    alpha_pixels = alpha_channel.getdata()
    total_pixels = len(alpha_pixels)
    trans_pixels = sum(1 for pixel in alpha_pixels if pixel > 128)
    return (trans_pixels / total_pixels) < threshold

PATH = r'D:\FAVORFIT\digital_art'
for file in os.listdir(PATH):
    if file.lower().endswith(('.png')):
        image_path = os.path.join(PATH, file)
        if transparent(image_path):
            os.remove(image_path)
            print(f"Removed {file}")
