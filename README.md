# Bridge Attention v2 (BAv2)

**BAv2** is an improved channel attention mechanism designed to enhance cross-layer information flow in convolutional neural networks. It introduces an adaptive selection operator to reduce redundancy and optimize feature interaction across layers. BAv2 achieves significant improvements on ImageNet classification benchmarks when integrated into ResNet and other advanced architectures.

---

## 🔍 Background

This work builds upon our previous method, **BA-Net: Bridge Attention for Deep Convolutional Neural Networks**, which proposed a novel bridge attention mechanism to facilitate feature integration across layers.

- 📖 [BA-Net Paper (ECCV, 2022)](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_18)  
- 💻 [BA-Net Code (BAv1)](https://github.com/zhaoy376/Bridge-Attention)

In BAv2, we further introduce an **adaptive selection operator** to filter redundant information and enhance cross-layer communication more effectively.

---

## 📌 Highlights

- Adaptive and lightweight attention mechanism
- Substantial performance boost on ImageNet (Top-1: 80.49% with ResNet50, 81.75% with ResNet101)
- Outperforms SENet and other classical channel attention modules
- Easily pluggable into various CNNs and vision transformers

---

## 📂 Usage

### Training

、、、
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 main.py --batch_size 256 --model_name ba_resnet50_v2 --data_path *path to your ImageNet dataset* --tag 300epochs_4gpu_256_imagenet
、、、

### Eval

、、、
python -m torch.distributed.launch --nproc_per_node 1 --master_port 13456 eval.py --batch_size 128 --model_name ba_resnet50_v2 --dataset ImageNet --resume ba_resnet50_v2.pth --data_path *path to your ImageNet dataset*
、、、

---

## 📬 Contact

For questions or collaborations, please contact: [chenjunzhou@mail.sysu.edu.cn; zourz@mail2.sysu.edu.cn]

