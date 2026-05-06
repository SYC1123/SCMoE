# SCMoE: Robust Medical Image Segmentation Across Clinical Domains



## Overview
SCMoE (Structural-shift-aware Mixture-of-Experts) is a novel framework for **multi-source domain generalization in medical image segmentation**. It addresses the challenges of **cross-domain variability** caused by different imaging devices and acquisition protocols. The framework combines multiple expert models via a Mixture-of-Experts (MoE) architecture, equipped with:

- **Dynamic Structural Attention (DSA)**: Adaptively models spatial structures across domains.
- **Hierarchical Gating Calibration (HGC)**: Combines expert predictions in an uncertainty-aware, pixel-level and expert-level calibrated manner.

SCMoE achieves **robust segmentation performance on unseen domains**, demonstrated on Fundus and Prostate datasets.

---

## Key Features
- **Multi-Source Domain Generalization**: Generalizes across images from different hospitals or devices without retraining.
- **Expert Specialization**: Each expert network learns domain-specific features for better adaptability.
- **Dynamic Structural Attention**: Learns spatial relationships adaptively to handle structural shifts.
- **Hierarchical Gating Calibration**: Stabilizes fusion of multiple experts by weighting their contributions according to prediction confidence.
- **Composite Loss Function**: Combines Dice loss and boundary loss to improve both region consistency and boundary accuracy.

---

## Installation
```bash
git clone https://github.com/yourusername/SCMoE.git
cd SCMoE
pip install -r requirements.txt
```

Requirements include:
- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- NumPy, OpenCV, scikit-learn, matplotlib  

---

## Results
**Prostate Dataset**  
- Average Dice Score: 85.92%  
- Average Surface Distance: 1.74 mm  

**Fundus Dataset**  
- Average Dice Score: 87.27%  
- Average Surface Distance: 12.60 mm  

**Ablation Study**  
- Removing **DSA** or **HGC** reduces Dice by 1.7% and 1.63%, respectively.  
- Both components together yield the best segmentation performance.  
(Refer to Table 3, pages 15-16, and Figure 2, page 14, for qualitative comparisons.)

---

## License
This project is licensed under the MIT License.
