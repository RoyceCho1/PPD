# Personalized Preference Fine-tuning of Diffusion Models

This is the public implemenation of [Personalized Preference Fine-tuning of Diffusion Models](https://arxiv.org/abs/2501.06655) (PPD). PPD is a comprehensive framework for personalized text-to-image generation. This project adapts from the [LLaVA-NeXT repository](https://github.com/LLaVA-VL/LLaVA-NeXT).

## 🚀 Overview

This project implements a personalized text-to-image generation system that learns individual user preferences through:

- **User Classification**: Training neural networks to identify users based on their image preference patterns
- **Multimodal Embeddings**: Leveraging LLaVA models to extract rich visual-textual representations
- **Preference Learning**: Understanding user aesthetics through preference pairs and feedback
- **Personalized Generation**: Adapting image generation to match individual user preferences

## 📁 Project Structure

```
personalized-t2i/
├── user_classification/        # User preference classification models
│   └── user_classifier.py     # Neural network classifier for user identification
├── llava_embeddings/          # LLaVA-based embedding generation
│   ├── pick_a_pick_user_emb.py    # Generate user embeddings from preference pairs
│   └── pick_a_pick_user_cond.py   # Conditional user embedding generation
├── eval/                      # Evaluation frameworks
│   ├── eval_winrate_gpt4o.py    # CGPT 4o-based win rate evaluation
│   ├── eval_winrate_gpt4o_userdesc.py  # User description evaluation
│   └── eval_winrate.py        # Standard win rate evaluation
├── scripts/                   # Shell scripts for automation
│   ├── run_user_classify.sh   # Run user classification training
│   ├── gen_emb.sh            # Generate embeddings
│   ├── eval_winrate_gpt4o.sh    # Run GPT 4o evaluations
│   ├── eval_winrate.sh       # Run standard evaluations
│   └── eval_winrate_gpt4o_userdesc.sh
├── requirements.txt           # Python dependencies
└── LICENSE                   # Apache 2.0 License
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd personalized-t2i
   ```

2. **Create and activate a virtual environment:**
   ```bash
   conda create -n personalized-t2i python=3.10 -y
   conda activate personalized-t2i
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[train]"
   ```

   Or install from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Key Components

### 1. User Classification (`user_classification/`)

The user classifier learns to identify users based on their image preference patterns using deep neural networks.

**Features:**
- Multi-layer perceptron with residual connections
- Layer normalization and dropout for regularization
- Support for top-k accuracy metrics
- WandB integration for experiment tracking

**Usage:**
```bash
./scripts/run_user_classify.sh
```

### 2. LLaVA Embeddings (`llava_embeddings/`)

Generates rich multimodal embeddings using state-of-the-art LLaVA models to understand user preferences.

**Key Features:**
- Integration with LLaVA-OneVision models
- Few-shot preference learning
- Multi-image processing capabilities
- User profile generation from preference pairs

**Usage:**
```bash
./scripts/gen_emb.sh
```

### 3. Evaluation Framework (`eval/`)

Comprehensive evaluation suite for measuring personalization performance.

**Evaluation Methods:**
- **Win Rate Evaluation**: Direct preference comparison
- **ChatGPT-based Evaluation**: LLM-assisted preference assessment
- **User Description Evaluation**: Text-based user profile evaluation

**Usage:**
```bash
# Standard win rate evaluation
./scripts/eval_winrate.sh

# GPT 4o based evaluation
./scripts/eval_winrate_chatgpt.sh

# User description evaluation
./scripts/eval_winrate_chatgpt_userdesc.sh
```

## 📊 Datasets

This project works with the [Pick-a-Pic dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2), which contains user preference annotations for image pairs.

## 🎨 Usage Examples

### Training a User Classifier

```bash
python user_classification/user_classifier.py \
    --dataset_name "your_dataset" \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-3 \
    --wandb_project "personalized_t2i"
```

### Generating User Embeddings

```bash
python llava_embeddings/pick_a_pick_user_emb.py \
    --num_shots 4 \
    --pretrained "lmms-lab/llava-onevision-qwen2-7b-ov-chat" \
    --output_dir "./data"
```

### Running Evaluations

```bash
python eval/eval_winrate_chatgpt.py \
    --dataset_name "your_test_dataset" \
    --model_name "gpt-4o-mini" \
    --include_cot
```

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LLaVA-NeXT Team**: This project adapts from the excellent [LLaVA-NeXT repository](https://github.com/LLaVA-VL/LLaVA-NeXT)
- **Pick-a-Pic Dataset**: Thanks to the creators of the Pick-a-Pic preference dataset
- **Hugging Face**: For providing the model hosting and dataset infrastructure
- **OpenAI**: For GPT models used in evaluation

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{dang2025personalizedpreferencefinetuningdiffusion,
      title={Personalized Preference Fine-tuning of Diffusion Models}, 
      author={Meihua Dang and Anikait Singh and Linqi Zhou and Stefano Ermon and Jiaming Song},
      year={2025},
      eprint={2501.06655},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.06655}, 
}
```

## 🔗 Related Work

- [LLaVA-NeXT: Open Large Multimodal Models](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2)
- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

---

For questions or support, please open an issue on GitHub or contact the maintainers.
