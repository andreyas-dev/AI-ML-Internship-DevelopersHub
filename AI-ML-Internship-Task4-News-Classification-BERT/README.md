# 📰 Task 4 – News Classification with BERT

This project is part of my **AI/ML Internship Task 4** and focuses on building a **news classification model** using a **pre-trained BERT transformer**.  
The goal is to classify news articles into **four categories** from the **AG News dataset** by fine-tuning `bert-base-uncased`.

---

## 🎯 Objectives
- 📥 **Load & Inspect** the AG News dataset  
- 📝 **Tokenize Text** using the BERT tokenizer  
- 🤖 **Fine-tune BERT** for news classification  
- 📊 **Evaluate Performance** using Accuracy and Weighted F1-Score  
- 💾 **Save Trained Model & Tokenizer** for inference  

---

## 📂 Folder Contents
- 📒 **news_classification_bert.ipynb** → Jupyter Notebook containing full implementation
- 📒 **README.md** 

---

## 🚀 How to Run
1. 📂 **Open Notebook:** Launch `news_classification_bert.ipynb` in **Jupyter Notebook**  
2. ▶️ **Run Cells:** Execute sequentially to load dataset, tokenize text, train BERT, and evaluate performance  
3. 👀 **Analyze Results:** Review:
   - 📊 **Model Metrics** — Accuracy, Weighted F1-Score  
   - 📈 **Training Plots** — Loss & metric curves per epoch  

---

## ⚙️ Training Configuration

| Hyperparameter       | Value           |
|---------------------|----------------|
| Learning Rate        | 2e-5           |
| Batch Size           | 16 (train & eval) |
| Epochs               | 3              |
| Weight Decay         | 0.01           |
| Max Sequence Length  | 128            |
| Evaluation           | Per epoch      |

---

## 📊 Evaluation Metrics
- **Accuracy** → Measures correct predictions  
- **Weighted F1-Score** → Handles class imbalance  
- **Best Model Selection** → Based on accuracy  

---

## 💾 Model Output
After training, the model and tokenizer are saved in:

./news_classifier/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt

These files can be easily reloaded for inference.

---

## 🛠 Tech Stack
- 🐍 **Language:** Python  
- 📚 **Libraries:** `transformers`, `torch`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`  

---

## 📫 Contact
- **LinkedIn:** [Andreyas](www.linkedin.com/in/eng-andreyas)  
- **Email:** eng.andreyas@gmail.com    

---

## ✅ Status
**Task Completed Successfully** – BERT fine-tuned, evaluated, and model files saved for reuse.

