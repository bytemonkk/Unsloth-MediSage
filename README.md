# Unsloth-MediSage
### Parameter-Efficient Fine-Tuning of Llama-3-8B-Instruct for Medical Question Answering using Unsloth and LoRA

<p align="center">
  <img src="assets/MedicalQA_Architecture.png" width="100%">
</p>

<h1 align="center">🩺 MediSage-Unsloth</h1>

<h3 align="center">
Fine-Tuning Llama-3-8B-Instruct for Medical Question Answering using Unsloth
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red">
  <img src="https://img.shields.io/badge/Llama--3--8B-Instruct-green">
  <img src="https://img.shields.io/badge/Unsloth-Fast%20Fine--Tuning-orange">
  <img src="https://img.shields.io/badge/LoRA-PEFT-yellow">
  <img src="https://img.shields.io/badge/Task-Medical%20Question%20Answering-purple">
</p>

---

# 📖 Overview

**MediSage-Unsloth** is a domain-adapted Medical Question Answering (Medical QA) system developed by fine-tuning **Llama-3-8B-Instruct** on a **Medical Q&A Dataset** using **Unsloth** and **Low-Rank Adaptation (LoRA)**. The project leverages parameter-efficient fine-tuning to create a lightweight medical assistant capable of generating context-aware and instruction-following responses for healthcare-related queries.

---

# ✨ Features

* 🩺 Medical Question Answering
* 🦙 Fine-tuned Llama-3-8B-Instruct
* ⚡ Fast Fine-Tuning with Unsloth
* 🎯 Parameter-Efficient Training using LoRA
* 📚 Instruction-Tuned Medical Dataset
* 💾 Low GPU Memory Consumption
* 🤖 Context-Aware Medical Responses
* 🚀 Ready for Hugging Face Deployment

---

# 🏗️ Model Architecture

```text
                 Medical Q&A Dataset
                         │
                         ▼
          Data Cleaning & Preprocessing
                         │
                         ▼
          Instruction Prompt Formatting
                         │
                         ▼
                 Tokenization
                         │
                         ▼
            Llama-3-8B-Instruct Model
                    (Base LLM)
                         │
                         ▼
          LoRA Adapter Injection (PEFT)
                         │
                         ▼
           Fine-Tuning using Unsloth
                         │
                         ▼
           Fine-Tuned Medical QA Model
                         │
                         ▼
              User Medical Question
                         │
                         ▼
         Context-Aware Medical Response
```

---

# ⚙️ Technology Stack

* Python
* PyTorch
* Hugging Face Transformers
* Unsloth
* PEFT (LoRA)
* TRL
* Accelerate
* BitsAndBytes
* Datasets

---

# 📂 Dataset

The model is trained using a **Medical Question & Answer Dataset** consisting of healthcare-related questions and expert-style answers formatted for instruction tuning.

### Example

**Input**

```text
What are the symptoms of asthma?
```

**Output**

```text
Common symptoms include wheezing, shortness of breath,
chest tightness, and persistent coughing.
```

---

# 🚀 Training Pipeline

```text
Medical Q&A Dataset
          │
          ▼
Data Cleaning
          │
          ▼
Instruction Formatting
          │
          ▼
Tokenizer
          │
          ▼
Llama-3-8B-Instruct
          │
          ▼
LoRA Adapters
          │
          ▼
Unsloth Fine-Tuning
          │
          ▼
Fine-Tuned Medical QA Model
```

---

# 💡 Applications

* Medical Question Answering
* AI Healthcare Assistant
* Patient Education
* Clinical Knowledge Support
* Healthcare Chatbots
* Medical Information Retrieval

---

# 📁 Project Structure

```bash
MediSage-Unsloth/
│
├── assets/
│   └── MedicalQA_Architecture.png
│
├── dataset/
│   └── medical_qa_dataset.csv
│
├── notebooks/
│   └── training.ipynb
│
├── inference.py
├── train.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🔮 Future Work

* Retrieval-Augmented Generation (RAG)
* Multi-turn Medical Conversations
* Medical Report Understanding
* Clinical Document Question Answering
* Deployment with FastAPI & Hugging Face Spaces

---

# ⭐ Star this repository if you found it useful!
