import numpy as np
from zennit.image import imsave
from zennit.image import CMAPS


# fname = "2008_000021"
# path = "../results/attributions/{}.npy".format(fname)
#
# img = np.load(path)
# # print(img.shape)
# # img.reshape((img.shape[2], img.shape[0], img.shape[1]))
# # img = np.sum(img, axis=2)[:, :, np.newaxis]
# img_lrp = np.sum(img, axis=2)
# amax = img_lrp.max((0, 1), keepdims=True)
# img_lrp = (img_lrp + amax) / 2 / amax
# imsave("../results/attributions/{}.png".format(fname), img_lrp, vmin=0., vmax=1., level=2., cmap="bwr")
# # print(img.shape)
# print(CMAPS)

classidx = 0
fname = "2008_000021"
# classidx = 1
# fname = "2008_000090"
xai_method = "epsilon_plus"

path = "../attributions/{}/{}/{}.npy".format(classidx, xai_method, fname)

img = np.load(path)

img_lrp = np.sum(img, axis=2)
amax = img_lrp.max((0, 1), keepdims=True)
img_lrp = (img_lrp + amax) / 2. / amax
imsave("../attributions/{}_{}.png".format(xai_method, fname), img_lrp, vmin=0., vmax=1., level=2., cmap="bwr")

path_2 = "../attributions/{}/{}/{}.npy".format(classidx, "epsilon_plus_flat", fname)

img2 = np.load(path_2)

equal = (img == img2).all()
print(equal)

