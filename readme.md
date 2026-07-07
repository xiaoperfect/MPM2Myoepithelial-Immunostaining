# MPM2Myoepithelial-Immunostaining

This repository provides the PyTorch implementation for label-free multiphoton microscopy (MPM)-based virtual myoepithelial immunostaining.

The model learns a paired image-to-image translation mapping from label-free MPM images to CK5/6 myoepithelial immunohistochemical images. It is designed to generate virtual myoepithelial immunostaining images for assisting the assessment of myoepithelial integrity in breast ductal carcinoma in situ with microinvasion.

## Overview

Conventional myoepithelial immunohistochemistry requires additional tissue sections, antibody incubation, chromogenic staining and multiple sample-processing steps. In this project, label-free MPM images are used as input, and a Pix2Pix-based conditional generative adversarial network is trained to generate virtual CK5/6 myoepithelial immunohistochemical images.

The task is formulated as paired image-to-image translation:

```text
Input A  : label-free MPM image
Target B : CK5/6 myoepithelial immunohistochemical image
Output   : virtual CK5/6 myoepithelial immunohistochemical image
```

## Repository structure

```text
MPM2Myoepithelial-Immunostaining/
├── data/
│   ├── trainA/
│   ├── trainB/
│   ├── testA/
│   └── testB/
├── datasets/
│   ├── combine_A_and_B.py
│   └── MPM_CK56/
│       ├── train/
│       └── test/
├── models/
├── options/
├── util/
├── train.py
├── test.py
├── evaluate.py
├── requirements.txt
└── readme.md
```

## Environment

This code was implemented using Python and PyTorch.

Install the required packages:

```bash
pip install -r requirements.txt
```

The typical environment includes:

```text
python
torch
torchvision
numpy
Pillow
opencv-python
dominate
visdom
scikit-image
matplotlib
tqdm
```

## Data preparation

The MPM images and CK5/6 immunohistochemical images should be spatially aligned before training.

For paired Pix2Pix training, each input MPM image should have a corresponding CK5/6 immunohistochemical image with the same filename.

The expected folder structure before combining paired images is:

```text
data/
├── trainA/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── trainB/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── testA/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── testB/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

Here:

```text
A: label-free MPM images
B: CK5/6 myoepithelial immunohistochemical images
```

The filenames in `trainA` and `trainB` should match. The same rule applies to `testA` and `testB`.

## Combine aligned image pairs

This repository follows the Pix2Pix paired-data format, where the input image and the target image are concatenated into a single image.

To generate the training set:

```bash
python datasets/combine_A_and_B.py \
  --fold_A ./data/trainA \
  --fold_B ./data/trainB \
  --fold_AB ./datasets/MPM_CK56/train
```

To generate the test set:

```bash
python datasets/combine_A_and_B.py \
  --fold_A ./data/testA \
  --fold_B ./data/testB \
  --fold_AB ./datasets/MPM_CK56/test
```

After this step, the dataset should be organized as:

```text
datasets/
└── MPM_CK56/
    ├── train/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── test/
        ├── 0001.png
        ├── 0002.png
        └── ...
```

Each image in `datasets/MPM_CK56/train` or `datasets/MPM_CK56/test` contains a paired image:

```text
left side  : MPM image
right side : CK5/6 immunohistochemical image
```

## Training

To train the Pix2Pix-based virtual immunostaining model:

```bash
python train.py \
  --dataroot ./datasets/MPM_CK56 \
  --name mpm_ck56_pix2pix \
  --model pix2pix \
  --direction AtoB \
  --dataset_mode aligned \
  --input_nc 3 \
  --output_nc 3
```

The trained checkpoints will be saved in:

```text
checkpoints/mpm_ck56_pix2pix/
```

## Testing

To generate virtual CK5/6 myoepithelial immunohistochemical images on the test set:

```bash
python test.py \
  --dataroot ./datasets/MPM_CK56 \
  --name mpm_ck56_pix2pix \
  --model pix2pix \
  --direction AtoB \
  --dataset_mode aligned \
  --input_nc 3 \
  --output_nc 3
```

The generated images will be saved in:

```text
results/mpm_ck56_pix2pix/
```

## Evaluation

Quantitative evaluation can be performed using:

```bash
python evaluate.py
```

The generated virtual immunostaining images are compared with the corresponding real CK5/6 immunohistochemical images.

The commonly used evaluation metrics include:

```text
PSNR
SSIM
```

Higher PSNR and SSIM values indicate better pixel-level reconstruction quality and structural similarity.

## Notes on image alignment

This repository assumes that the MPM and CK5/6 immunohistochemical images have already been spatially aligned before training. The model does not perform image registration internally.

Before running `combine_A_and_B.py`, please ensure that:

```text
1. Each MPM image has a corresponding CK5/6 image.
2. The paired images have the same filename.
3. The paired images have the same spatial size.
4. The paired images are already registered.
```

## Citation

If you use this code, dataset or trained model weights, please cite the corresponding paper:

```text
Label-free multiphoton microscopy-based virtual myoepithelial immunostaining for identifying breast ductal carcinoma in situ with microinvasion.
```

## Code and model weights

The analysis pipeline code and trained model weights supporting the findings of this study are publicly available in this GitHub repository.

## License

Please refer to the license information provided in this repository.
