import datasets
import pandas as pd
from datasets import load_dataset
import torch

_VERSION = datasets.Version("1.0.1")
METADATA_PATH = r"C:\Users\mlfav\lib\shlee\color_model\ColorModel\train.jsonl"
_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

class ColorModel(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="temp",
            features=datasets.Features(
                {
                    "RGB": datasets.features.Sequence(datasets.features.Value("float32")),
                    "HSV": datasets.features.Sequence(datasets.features.Value("float32")),
                    "CIELAB": datasets.features.Sequence(datasets.features.Value("float32")),
                    "RGB_feature": datasets.features.Sequence(datasets.features.Value("float32")),
                    "HSV_feature": datasets.features.Sequence(datasets.features.Value("float32")),
                    "LAB_feature": datasets.features.Sequence(datasets.features.Value("float32")),
                    "Hue_probability": datasets.features.Sequence(datasets.features.Value("float32")),
                    "Hue_Log_probability": datasets.features.Sequence(datasets.features.Value("float32")),
                    "Hue_adjacent_probability": datasets.features.Sequence(datasets.features.Value("float32")),
                    "Hue_Log_adjacent_probability": datasets.features.Sequence(datasets.features.Value("float32")),
                    "Hue_entropy": datasets.features.Sequence(datasets.features.Value("float32")),
                    "CH": datasets.features.Sequence(datasets.features.Value("float32")),
                    "Light_gradient": datasets.features.Sequence(datasets.features.Value("float32")),
                    "Hue_gradient": datasets.features.Sequence(datasets.features.Value("float32")),
                    "input_data": datasets.features.Sequence(datasets.features.Value("float32")),
                    "output_color": datasets.features.Sequence(datasets.features.Value("float32")),
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
            RGB = data["RGB"]
            HSV = data["HSV"]
            CIELAB = data["CIELAB"]
            RGB_feature = data["RGB_feature"]
            HSV_feature = data["HSV_feature"]
            LAB_feature = data["LAB_feature"]
            Hue_probability = data["Hue_probability"]
            Hue_Log_probability = data["Hue_Log_probability"]
            Hue_adjacent_probability = data["Hue_adjacent_probability"]
            Hue_Log_adjacent_probability = data["Hue_Log_adjacent_probability"]
            Hue_entropy = data["Hue_entropy"]
            CH = data["CH"]
            Light_gradient = data["Light_gradient"]
            Hue_gradient = data["Hue_gradient"]
            input_data = data["input_data"]
            output_color = data["output_color"]

            yield idx, {
                "RGB": torch.flatten(torch.tensor(RGB, dtype=torch.float32)),
                "HSV": torch.flatten(torch.tensor(HSV, dtype=torch.float32)),
                "CIELAB": torch.flatten(torch.tensor(CIELAB, dtype=torch.float32)),
                "RGB_feature": torch.flatten(torch.tensor(RGB_feature, dtype=torch.float32)),
                "HSV_feature": torch.flatten(torch.tensor(HSV_feature, dtype=torch.float32)),
                "LAB_feature": torch.flatten(torch.tensor(LAB_feature, dtype=torch.float32)),
                "Hue_probability": torch.flatten(torch.tensor(Hue_probability, dtype=torch.float32)),
                "Hue_Log_probability": torch.flatten(torch.tensor(Hue_Log_probability, dtype=torch.float32)),
                "Hue_adjacent_probability": torch.flatten(torch.tensor(Hue_adjacent_probability, dtype=torch.float32)),
                "Hue_Log_adjacent_probability": torch.flatten(torch.tensor(Hue_Log_adjacent_probability, dtype=torch.float32)),
                "Hue_entropy": torch.flatten(torch.tensor(Hue_entropy, dtype=torch.float32)),
                "CH": torch.flatten(torch.tensor(CH, dtype=torch.float32)),
                "Light_gradient": torch.flatten(torch.tensor(Light_gradient, dtype=torch.float32)),
                "Hue_gradient": torch.flatten(torch.tensor(Hue_gradient, dtype=torch.float32)),
                "input_data": torch.flatten(torch.tensor(input_data, dtype=torch.float32)),
                "output_color": torch.flatten(torch.tensor(output_color, dtype=torch.float32)),
            }

if __name__ == "__main__":
    dataset = load_dataset(path=r"C:\Users\mlfav\lib\shlee\color_model\ColorModel")
    dataset = dataset["train"]
    print(dataset.shape)
    print(dataset[0])
    print(len(dataset[0]))
    