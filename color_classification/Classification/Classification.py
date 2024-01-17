import datasets
import pandas as pd
from datasets import load_dataset
import torch

_VERSION = datasets.Version("0.0.1")
METADATA_PATH = r"/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/jsonl/quantized_product_colors_4to1.jsonl"
_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

class Classification(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="temp",
            features=datasets.Features(
                {
                    "input_colors": datasets.features.Sequence(datasets.features.Value("float32")),
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
            input_colors = data["obj_colors"]
            output_color = data["bg_color"]

            yield idx, {
                "input_colors": torch.flatten(torch.tensor(input_colors, dtype=torch.int32)),
                "output_color": torch.flatten(torch.tensor(output_color, dtype=torch.int32)),
            }

if __name__ == "__main__":
    dataset = load_dataset(path=r"/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/Classification")
    dataset = dataset["train"]
    print(dataset.shape)
    print(dataset[0])
    print(len(dataset[0]))
    