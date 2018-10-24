# Kaggle-tgs
UNet-based for segmenting salt deposits from seismic images.

# Genearal
Recently, I have participated in Kaggle competition [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) and reached 19th place (Silver medal). This repository contains a simplified and clean up version of my code.

I used UNet-based architecture with resnet-style encoder (resnet34, SE-ResNeXt50, SE-ResNeXt101) and decoder with [scSE](https://arxiv.org/pdf/1803.02579.pdf). The best is SE-ResNeXt50 which get LB(0.887) within 10 folds. Finally, the private LB 0.8898 was achieved without post-propocess.

# Requirements
python3

pytorch >= 0.4

cv2

numpy

matploitlib

# Training
Train: python3 train.py

test:  python3 submit.py


# Aknowledgement
Thanks to [Heng](https://www.kaggle.com/hengck23) who is really a thoughtful man. I learned a lot from his kaggle posts. Without his idea, it would be difficult for me to reach a top 20 place.

# Related paper
[Resnet V1](https://arxiv.org/pdf/1512.03385.pdf)

[Resnet V2](https://arxiv.org/pdf/1603.05027.pdf)

[ResNeXt](https://arxiv.org/pdf/1611.05431.pdf)

[SENet](https://arxiv.org/pdf/1709.01507.pdf)

[Hypercolumn](https://arxiv.org/pdf/1411.5752.pdf)

[Deeply Supervised Learning](https://arxiv.org/pdf/1708.01241.pdf)


