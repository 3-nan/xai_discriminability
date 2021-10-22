import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("/home/motzkus/work/xai_discriminability/separability")
from xaitestframework.dataloading.custom import VOC2012Dataset, MyImagenetDataset
from xaitestframework.dataloading.dataloader import DataLoader

classindices = [96, 126, 155, 292, 301, 347, 387, 405, 417, 426,
                446, 546, 565, 573, 604, 758, 844, 890, 937, 954]

results = []

for classidx in classindices:
    # initialize dataset and dataloader
    print(os.path.isdir("../../data/VOC2012"))
    dataset = MyImagenetDataset("../../data/imagenet/imagenet", "val", classidx=[classidx])
    # dataset = VOC2012Dataset("../../data/VOC2012/", "val", classidx=["0"])
    dataset.set_mode("binary_mask")

    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    print("Class name is {}".format(dataset.cmap[int(classidx)]))

    for b, batch in enumerate(dataloader):

        for sample in batch:

            binary_mask = sample.binary_mask[dataset.cmap[int(classidx)]]

            if len(binary_mask.shape) != 3:
                print("shape mismatch")

            binary_mask = binary_mask.astype(bool)[:, :, 0]

            ratio = np.float(np.sum(binary_mask)) / (binary_mask.shape[0]*binary_mask.shape[1])

            results.append(ratio)

print(len(results))
print("Mean object size ratio is {}".format(np.mean(results)))

plt.figure()
weights = np.ones_like(results) / len(results)
plt.hist(results, bins=3, weights=weights, rwidth=0.8)
plt.show()
