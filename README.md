
<div align="center">
  <h1>Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving</h1>
  <!-- <h3>Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving</h3> -->
  <p>Shuai Liu, Quanmin Liang, Zefeng Li, Boyang Li, Kai Huang</p>
  <p>School of Computer Science and Engineering, Sun Yat-sen University</p>
  <p><i>Contact: liush376@mail2.sysu.edu.cn, liby83@mail.sysu.edu.cn, huangk36@mail.sysu.edu.cn</i></p>
</div>

---

## 📰 News

- `Sep 18, 2025`: GaussianFusion has been accepted to NeurIPS 2025 (Spotlight).
- `Oct 09, 2025`: The source code of GaussianFusion has been publicly released.

---

## Table of Contents

- [Introduction](#-Introduction)
- [Results on NAVSIM](#-Results-on-NAVSIM)
- [Video Demo](#-Video-Demo)
- [Getting Started](#-Getting-Started)
- [Citation](#-Citation)
- [Acknowledgement](#-Acknowledgement)
- [Contact](#-Contact)

---

## 🌟 Introduction

End-to-end (E2E) autonomous driving aims to simplify the traditional modular pipeline by directly mapping multi-sensor inputs to driving actions through deep learning, enabling joint optimization and reducing system complexity. However, relying on a single sensor limits robustness in diverse scenarios, making multi-sensor fusion essential for reliable perception and robust driving policies.

Existing fusion strategies are mainly categorized as flatten fusion and bird’s eye view (BEV) fusion. Flatten fusion compresses sensor features into a shared latent space using attention mechanisms, offering flexibility but lacking explicit spatial grounding. BEV fusion projects multi-modal features into a common BEV coordinate, improving spatial understanding but incurring high computational and memory costs due to dense representations.

GaussianFusion introduces a novel Gaussian-based multi-sensor fusion framework for E2E autonomous driving. Our approach leverages sparse and interpretable 2D Gaussians to efficiently integrate multi-modal features from cameras and LiDARs. The dual-branch fusion pipeline is tailored for both scene reconstruction and motion planning, requiring only BEV semantic supervision. A cascade planning module further refines trajectory predictions through hierarchical Gaussian queries.

Extensive evaluations on both open-loop (NAVSIM) and closed-loop (Bench2Drive) benchmarks demonstrate the superior performance and robustness of GaussianFusion compared.

<div align="center">
<b>The overall framework of our GaussianFusion.</b><br>
<img src="assets\overall framework.png" />
</div>



---

## 📊 Results on NAVSIM

<div align="center">
<b>Visualization of Gaussians during the refinement process.</b><br>
<img src="assets\Visualization of Gaussians.png" /><br><br>
<b>Visualization of predicted and ground-truth trajectories</b><br>
<img src="assets\Visualization of predicted.png" />
</div>



---

## 🖼️ Video Demo
<div align="center">
<b>Demo of GaussianFusion (GIF)</b><br>
<img src="assets/gaussianfusion_demo.gif" alt="GaussianFusion demo GIF" />
<br><br>
</div>


---

## 🧰 Getting Started

### 1. Environment Preparation

- [Getting started from NAVSIM environment preparation](https://github.com/autonomousvision/navsim?tab=readme-ov-file#getting-started-)
- [Preparation of GaussianFusion environment](docs/install.md)

### 2. Training

Before starting training, ensure that the dataset cache is already prepared and the path settings are correct.

```bash
bash ./scripts/training/run_gaussianfusion.sh 
```

### 3. Evaluation

Before starting evaluation, ensure that the metric cache is already prepared and the path of pt file is correct.

```bash
bash ./scripts/evaluation/run_gaussianfusion.sh
```

---

## 📜 Citation

If any part of our paper or code is helpful to your research, please consider citing our work 📝 and give us a star ⭐. Thanks for your support!

```bibtex
@article{liu2025gaussianfusion,
  title={GaussianFusion: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving},
  author={Liu, Shuai and Liang, Quanmin and Li, Zefeng and Li, Boyang and Huang, Kai},
  journal={arXiv preprint arXiv:2506.00034},
  year={2025}
}
```

---

## ❤️ Acknowledgement
This is an enhanced implementation of GaussianFusion which is slightly different from the paper version. The code is built upon the great open-source efforts of:

- [NAVSIM](https://github.com/autonomousvision/navsim)
- [TransFuser](https://github.com/autonomousvision/transfuser)
- [GaussianFormer](https://github.com/GaussianFormer)
- [iPad](https://github.com/Kguo-cs/iPad/tree/main)
---

## 📫 Contact

For questions, feel free to reach out:
📧 liush376@mail2.sysu.edu.cn / liby83@mail.sysu.edu.cn /
