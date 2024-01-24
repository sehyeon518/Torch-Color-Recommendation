import numpy as np
import json
from models.model import forward


def run(input_data):
    with open("favorfit_color_recommendation/models/weight_and_bias.json") as rg:
        model_weights = json.load(rg)

    for key in model_weights:
        model_weights[key] = np.array(model_weights[key])

    # 모델 연산 수행
    output = forward(input_data, model_weights)
    return output


if __name__ == "__main__":
    run()
