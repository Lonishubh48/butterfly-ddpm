# 🦋 Butterfly Image Generation with DDPM

A from-scratch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** trained on the [Smithsonian Butterflies dataset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) using HuggingFace Diffusers and PyTorch.

> 📖 This project follows concepts from *Using Stable Diffusion with Python* by Andrew Zhu — implemented and experimented with independently on Google Colab.

---

## 🖼️ What This Project Does

This notebook trains a diffusion model that **learns to generate butterfly images from pure Gaussian noise**, step by step, by reversing a noising process.

The pipeline:
1. Takes real butterfly images
2. Gradually adds noise over 1000 timesteps (forward diffusion)
3. Trains a UNet to predict and remove that noise
4. At inference, starts from random noise and denoises back into a butterfly image

---

## 📁 Project Structure

```
butterfly-ddpm/
│
├── Diffusion_model.ipynb   # Main training & inference notebook
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local GPU with CUDA

### Installation

```bash
pip install diffusers transformers accelerate torch datasets torchvision matplotlib
pip install triton
```

### Run the Notebook

Open `Diffusion_model.ipynb` in Google Colab and run all cells top to bottom.

> ⚠️ A **GPU runtime** is required. In Colab: `Runtime → Change runtime type → T4 GPU`

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Model | UNet2D with Attention |
| Input size | 64 × 64 × 3 |
| Block channels | 64 → 128 → 256 → 512 |
| Down blocks | `DownBlock2D`, `DownBlock2D`, `AttnDownBlock2D`, `AttnDownBlock2D` |
| Up blocks | `AttnUpBlock2D`, `AttnUpBlock2D`, `UpBlock2D`, `UpBlock2D` |
| Scheduler | DDPMScheduler (1000 timesteps) |
| Beta range | 0.001 → 0.02 (linear) |

---

## 🏋️ Training

| Parameter | Value |
|---|---|
| Dataset | `huggan/smithsonian_butterflies_subset` |
| Image size | 64 × 64 |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Loss function | MSE (predicted noise vs actual noise) |

---

## 🔬 Inference

Two inference methods are implemented:

**1. DDPMPipeline (simple)**
```python
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
images = pipeline(batch_size=4, num_inference_steps=50).images
```

**2. Manual Reverse Diffusion Loop**
Denoises step by step and saves intermediate outputs every 10 steps — great for visualizing the refinement process.

---

## 📊 Results

- Training loss curve tracked over all steps
- Generated images visualized after 50 epochs at 64×64 resolution
- Intermediate denoising steps visualized (noise → butterfly)

---

## 💡 What I Learned

- How the **forward diffusion process** adds noise mathematically: `q(x_t | x_0)`
- How a **UNet predicts noise** at each timestep rather than predicting the image directly
- The role of **attention blocks** in capturing global structure
- How **reverse diffusion** iteratively denoises from `x_T → x_0`
- Hands-on use of HuggingFace `diffusers` library

---

## 🔭 Future Experiments

- [ ] Try cosine beta schedule instead of linear
- [ ] Train for 100–200 epochs for sharper results
- [ ] Increase image size to 128×128
- [ ] Add conditional generation (by butterfly species)
- [ ] Compute FID score to measure generation quality

---

## 📚 References

- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
- [DDPM Paper — Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- [Smithsonian Butterflies Dataset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset)
- *Using Stable Diffusion with Python* — Andrew Zhu

---

## 👤 Author

Learning ML one experiment at a time 🚀  
Feel free to fork, star ⭐, or open an issue if you have ideas!
