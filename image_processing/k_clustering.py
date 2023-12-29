from PIL import Image
import numpy as np
import os
import json

old_settings = np.seterr(divide='ignore', invalid='ignore')

def kmeans(image_array, k=8, iteration=5):
    pixels = image_array.reshape(-1, 3)
    centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]

    for _ in range(iteration):
        with np.errstate(invalid='ignore'):
            distance = np.linalg.norm(pixels - centroids[:, np.newaxis], axis=2)
            labels = np.argmin(distance, axis=0)

            new_centroids = np.array([pixels[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else np.zeros(pixels.shape[1]) for i in range(k)])
            if np.all(new_centroids == centroids):
                break
            centroids = new_centroids

    return centroids.astype(int)


def extract_colors(image_path, colors=8):
    image = Image.open(image_path)
    image = image.convert("RGB")

    image_array = np.array(image)

    colors = kmeans(image_array, k=colors)
    np.seterr(**old_settings)

    colors = np.clip(colors, 0, 255)

    return colors


data = {"data": []}

def save_json(train_colors, label_colors):
    data_entry = {"input": train_colors, "output": label_colors}
    data["data"].append(data_entry)


PATH = r'D:\FAVORFIT\digital_art\digital_art'

count = 0
for file in os.listdir(PATH):
    image_path = os.path.join(PATH, file)
    colors = extract_colors(image_path)

    train_colors = colors[:4].tolist()
    label_colors = colors[4:].tolist()
    save_json(train_colors, label_colors)
    count += 1
    if count % 1000 == 0:
        print(count)

print("done:", count)

with open('Data\Color_Data\data.json', "w") as file:
    json.dump(data, file, indent=2)
