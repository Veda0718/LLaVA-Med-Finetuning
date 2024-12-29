# LLaVA-Med Fine-Tuning and Demo

This repository provides resources to fine-tune and deploy LLaVA-Med, a state-of-the-art vision-language model for medical imaging tasks. It includes scripts for fine-tuning the model, testing its performance, and launching an interactive Gradio-based demo for radiologists and researchers.

To get started, clone the LLaVA-Med repository and navigate to LLaVA-Med folder

```
git clone https://github.com/onyekaokonji/LLaVA-Med.git
cd LLaVA-Med
```

 - **_Finetuning.ipynb_**: Notebook for fine-tuning LLaVA-Med on domain-specific datasets such as VQA-RAD, SLAKE and MED-VQA.
 - **_app.py_**: Script to deploy a Gradio-based demo allowing users to interact with the fine-tuned model.

Note: To load LLaVA-Med with 4-bit quantization for the demo, ensure you have the latest version of bitsandbytes. You can [download it here](https://pypi.org/project/bitsandbytes/#files) or upgrade your existing installation. 

## Model and Demo Hosting
 - The fine-tuned version of LLaVA-Med is hosted on [Hugging Face](https://huggingface.co/Veda0718/llava-med-v1.5-mistral-7b-finetuned).
 - Explore the live demo on Hugging Face Spaces [here](https://huggingface.co/spaces/Veda0718/Llava-Med).

For detailed experimental results on fine-tuning LLaVA-Med with VQA-RAD, SLAKE, and MED-VQA, refer to the _**report.pdf**_ included in this repository.

---
title: Llava Med
emoji: 🦀
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.8.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
