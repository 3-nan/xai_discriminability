import os
import sys
import numpy as np

sys.path.append("/home/motzkus/work/xai_discriminability/separability")
from xaitestframework.models.pytorchmodel import PytorchModel
from xaitestframework.experiments.model_parameter_randomization import spearman_distance

setting = "vgg16bn_imagenet"

# init model
model_path = "../models/pytorch/vgg16bn_imagenet/model.pt"
model_name = "vgg16bn"
model = PytorchModel(model_path, model_name)
print("Model loaded.")

explanation_dir = "../results/{}/mpr_example".format(setting)

# get layers including weights and iterate them
layer_names = model.get_layer_names(with_weights_only=True)

scores = []

for i in range(5):

    original_with_canon = np.load(os.path.join(explanation_dir, "{}_original.npy".format(i)))
    original_without_canon = np.load(os.path.join(explanation_dir, "{}_original_wbn.npy".format(i)))

    for layer_name in layer_names:
        with_canonization = np.load(os.path.join(explanation_dir, "{}_{}.npy".format(i, layer_name)))
        without_canonization = np.load(os.path.join(explanation_dir, "{}_{}_wbn.npy".format(i, layer_name)))

        if (with_canonization == without_canonization).all():
            print("No differences found.")

        # pure_score = np.abs(spearman_distance(original_with_canon, with_canonization))
        # print("Pure Score with canon: {}".format(pure_score))
        # pure_score = np.abs(spearman_distance(original_without_canon, without_canonization))
        # print("Pure Score without canon: {}".format(pure_score))

        # normalize explanations
        original_with_canon = original_with_canon / np.max(np.abs(original_with_canon))
        with_canonization = with_canonization / np.max(np.abs(with_canonization))

        original_without_canon = original_without_canon / np.max(np.abs(original_without_canon))
        without_canonization = without_canonization / np.max(np.abs(without_canonization))

        # compute distance value and append
        score = np.abs(spearman_distance(original_with_canon, with_canonization))
        # print("Final Score: {}".format(score))
        score_without_canon = np.abs(spearman_distance(original_without_canon, without_canonization))
        # print("Final Score without canon: {}".format(score_without_canon))

        print(score - score_without_canon)
        scores.append(score-score_without_canon)

print("sum is {}".format(np.sum(scores)))

