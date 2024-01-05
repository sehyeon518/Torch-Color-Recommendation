from PIL import Image
import datasets
import pandas as pd
from datasets import load_dataset
import torch

_VERSION = datasets.Version("0.0.2")
METADATA_PATH = r"C:\Users\mlfav\lib\shlee\color_palette\train.jsonl"
_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

class ColorPalette(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="temp",
            features=datasets.Features(
                {
                    "image": datasets.features.Value("string"),
                    "input_colors": datasets.features.Sequence(datasets.features.Value("int64")),
                    "output_colors": datasets.features.Sequence(datasets.features.Value("int64")),
                }
            ),
        )

    def _split_generators(self, dl_manager=None):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'data_list': METADATA_PATH}
            ),
        ]

    def _generate_examples(self, data_list):
        data_list = pd.read_json(data_list, lines=True)
        
        for idx, data in data_list.iterrows():
            img_path = data["image"]
            input_colors = data["input_colors"]
            output_colors = data["output_colors"]

            # img = Image.open(img_path).convert("RGB")
            # img = img.resize((256, 256))

            yield idx, {
                'image': img_path,
                'input_colors': torch.flatten(torch.tensor(input_colors, dtype=torch.float32)),
                'output_colors': torch.flatten(torch.tensor(output_colors, dtype=torch.float32)),
            }

if __name__ == "__main__":
    dataset = load_dataset(path=r"C:\Users\mlfav\lib\shlee\color_palette")
    dataset = dataset["train"] # dataset.data.shape = (100, 3)
    print(dataset.shape)
    print(type(dataset[0]['input_colors']))
    
