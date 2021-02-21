# censorface-js
Censorface in Javascript by ONNX.js. In this repository, it is the example of Retinaface which is a state-of-the-art single stage face detection algorithm shown in CVPR 2020.
The paper you can follow this [paper](https://arxiv.org/abs/1905.00641) which has title RetinaFace: Single-stage Dense Face Localisation in the Wild.

The Model is from [Pytorch model at this repository](https://github.com/biubug6/Pytorch_Retinaface) converted into ONNX model.

### Example
For demo the application, you can clone this repo in order to test the face detection algorithm by censoring the face by
using Retinaface with MobileNetV2 as a backbone and applying the two facial landmarks as anchor for the black box.

How to run this repo. You can run by type below.

```
yarn install
yarn start
```

### Result
Here is the result of the Censorface below.

![Censor result](https://github.com/nickuntitled/censorface-js/blob/censorface/result.png?raw=true "Censor Result")

```
Citation
=========
Deng J, Guo J, Zhou Y, Yu J, Kotsia I, Zafeiriou S. Retinaface: Single-stage dense face localisation in the wild. arXiv preprint arXiv:1905.00641. 2019 May 2.
```
