# Corneal Confocal Microscopy (CCM) Image Registration
This repository provides the official PyTorch implementation of the paper:

**Addressing Large Misalignment of Corneal Confocal Microscopy Image Frames with an Iterative Step-Aware Transformer Registration Network**
Zihao Chen, Zane Z. Zemborain, Raul E. Ruiz-Lozano, Manuel E. Quiroga-Garza, Symon Ma, Matias Soifer, Nadim S. Azar, Hazem M. Mousa, Victor L. Perez, and Sina Farsiu
***Biomedical Optics Express (BOE), 2026***

---

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/ZihaoChen0319/CCM_Registration
cd CCM_Registration
pip install -r requirements.txt
```

---

## Usage

### Train
To train the ISATR-Net:
```bash
python train.py
```
Training configurations (e.g., learning rate, batch size, dataset paths) can be modified directly in `train.py`.

### Evaluation
To evaluate a trained model:
```bash
python evaluate.py
```

### Using Your Own Data
#### Custom Dataset Class
You may use your own dataset by implementing a custom `Dataset` class, as long as the dataloader returns a dictionary with the following keys:
- `image_fixed` — fixed/reference image
- `image_moving` — moving image to be registered
- `keypoints_fixed` — keypoints in the fixed image
- `keypoints_moving` — corresponding keypoints in the moving image
- `num_keypoints` — number of valid keypoints

#### Use Provided Data Loader
A reference dataset implementation is provided in `data/`.
- Each image must be stored in `.npy` or `.npz` format. Each `.npy/.npz` file should contain an image, corresponding landmarks and the number of keypoints.
- A dataset JSON file must contain image pairs for: training, (optional) validation, and testing.

---

## Citation
If you find this work useful in your research, please cite:
```bibtex
@article{chen2026ccm,
  title={Addressing Large Misalignment of Corneal Confocal Microscopy Image Frames with an Iterative Step-Aware Transformer Registration Network},
  author={Chen, Zihao and Zemborain, Zane Z. and Ruiz-Lozano, Raul E. and Quiroga-Garza, Manuel E. and Ma, Symon and Soifer, Matias and Azar, Nadim S. and Mousa, Hazem M. and Perez, Victor L. and Farsiu, Sina},
  journal={Biomedical Optics Express},
  year={2026}
}
```



