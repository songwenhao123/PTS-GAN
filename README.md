# Progressive Text-Semantic-Aware Generative Adversarial Network for Image Fusion (PTS-GAN)

This is the official implementation of paper Progressive Text-Semantic-Aware Generative Adversarial Network for Image Fusion". 

### Abstract
*Infrared and visible image fusion (IVF) aims to synthesize comprehensive representations that preserve thermal signatures and visible textures. Existing IVF methods predominantly focus on pixel-level feature combinations but struggle to maintain semantic coherence in complex scenarios.
To address this challenge, we propose a Progressive Text-Semantic-Aware Generative Adversarial Network (PTS-GAN) for infrared and visible image fusion.
Specifically, we present a semantic-aware generator to preserve multi-scale local-global features of cross-modal semantics. It integrates the Dual Attention Routing (DAR) module with the Transformer architecture.Meanwhile, we propose a Textual Semantic Alignment (TSA) module to align CLIP text embeddings with multi-scale visual features. Moreover, a dual progressive discriminator is built to maintain semantic consistency between fused and source images through hierarchical adversarial training. Comprehensive experiments demonstrate that the proposed model outperforms the state-of-the-art methods objectively and subjectively.*

### Summary figure

<p align="center">
<img src="Figs/framework.png" alt="figure1"/>
</p>

## Code
### Install dependencies

```
# install cuda
Recommended cuda11.1

# create conda environment
conda create -n pts-gan python=3.9.12
conda activate pts-gan

# select pytorch version (recommended torch 1.8.2)
pip install -r requirements.txt
```

### To test
For the RGB and infrared image fusion:
```
python main_test_rgb_ir.py
```
