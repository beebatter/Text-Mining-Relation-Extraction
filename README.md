# **Relation Extraction Project (COMP61332 Coursework)**  

## **1. Project Overview**  
This project is part of the **COMP61332 Text Mining coursework**, focusing on **Relation Extraction (RE)**. The objective is to develop and evaluate **two different RE methods** using a **well-annotated dataset**. We compare a **traditional machine learning model (SVM)** with a **deep learning-based approach (BERT)**, implementing various enhancements to improve performance.

## **2. Dataset**  
We use the **SemEval-2010 Task 8** dataset, available at:  
[SemEval-2010 Task 8 Dataset](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8)  

### **Dataset Overview**  
- **Task**: Multi-way classification of semantic relationships between **nominal pairs**  
- **Training Samples**: **8000**  
- **Test Samples**: **2717**  
- **Relations**: **18 classes**, including `"Component-Whole(e2,e1)"`, `"Cause-Effect(e1,e2)"`, and `"Other"`  
- **Example Entry**:  
  - **Sentence**:  
    *The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.*  
  - **Relation**: *Component-Whole(e2,e1)*  

This dataset was chosen because it provides **clearly annotated entity relationships**, a **diverse set of relation types**, and a **balanced amount of training and test samples**, making it ideal for evaluating RE models.

## **3. Methods Implemented**  
We implemented two different relation extraction approaches:

### **Method 1: Support Vector Machine (SVM)**  
- A **traditional machine learning approach** using SVM as the classifier.  
- **Enhancements applied**:  
  - **TF-IDF vectorization** for better text representation.  
  - **BiLSTM + Attention** for contextual feature extraction.  
  - **GloVe embeddings** to enrich semantic understanding.  

### **Method 2: BERT-based RE Model**  
- A **deep learning-based approach** using **BERT (Bidirectional Encoder Representations from Transformers)**.  
- **Enhancements applied**:  
  - **Contrastive learning** to refine relation embeddings.  
  - **External knowledge augmentation** using:  
    - **WordNet** (for entity-level enhancement).  
    - **Wikidata** (for relation-level enrichment).  
  - **Learning rate scheduling** to improve training stability.  

## **4. Installation and Dependencies**  
The project includes **four Jupyter notebooks**, each corresponding to a specific model implementation:

| Notebook | Description |  
|----------|------------|  
| `original_bert.ipynb` | Baseline BERT model without enhancements. |  
| `data_enhanced.ipynb` | BERT model with **external knowledge augmentation**. |  
| `improved_bert.ipynb` | **Final optimized BERT model** with all enhancements. |  
| `SVM.ipynb` | Contains all **SVM versions**, from baseline to improved models. |  
| `BERT_evaluation.ipynb` | Evaluates the **best BERT model**. |  
| `SVM_evaluation.ipynb` | Evaluates the **best SVM model**. |  

### **Dependencies**  
Install all required dependencies using:  
```bash
pip install -r requirements.txt
```  
Alternatively, manually install key libraries:  
```bash
pip install pandas numpy scikit-learn torch transformers nltk tqdm
```  

### **Required Files**  
All necessary **datasets, trained models, and scripts** can be accessed here:  
https://drive.google.com/drive/folders/1WAryE-SpUyVgOM79f24LfW-GTpFnM4Rf?usp=sharing 

The shared folder contains:  
- **`improved_bert/`**:  
  - Enhanced **training & testing datasets (CSV format)**.  
  - Trained **BERT model (`improved_bert_relation_model/`)**.  
- **`original_bert/`**:  
  - Trained **baseline BERT model (`original_bert_relation_model/`)**.  
- **`SVM/`**:  
  - All **SVM-related data and trained models**.  

## **5. Running the Code**  
### **Running the Notebooks**  
1. **Upload the necessary files** from the **corresponding folder** (`improved_bert/`, `original_bert/`, or `SVM/`) to the **root directory** in Google Colab.  
2. Open the required notebook (`original_bert.ipynb`, `data_enhanced.ipynb`, `improved_bert.ipynb`, `SVM.ipynb`).  
3. **Run all cells sequentially** to execute:  
   - **Data preprocessing**  
   - **Model training**  
   - **Evaluation and inference**  
   - **Saving the trained models**  
   - **User-input relation extraction function**  

### **Running Inference**
After training, use the **pre-trained models** for **real-time relation extraction**:

```bash
python inference.py --model [model_name] --input "The <e1>company</e1> acquired the <e2>startup</e2>."
```

Alternatively, use the **evaluation notebooks** to analyze model performance:  
- `BERT_evaluation.ipynb` (BERT model evaluation)  
- `SVM_evaluation.ipynb` (SVM model evaluation)  

## **6. Evaluation**  
### **Performance Comparison**  
#### **SVM Results**  
- **Baseline SVM**:  
  - **Accuracy**: **53.77%**  
  - **F1-score**: **46.06%**  
- **Improved SVM (TF-IDF + BiLSTM + WordNet/GloVe)**:  
  - **Accuracy**: **70.30%**  
  - **F1-score**: **66.80%**  

#### **BERT Results**  
- **Baseline BERT**:  
  - **Accuracy**: **76.11%**  
  - **F1-score**: **71.24%**  
- **Improved BERT (Knowledge Augmentation + Contrastive Learning + LR Scheduling)**:  
  - **Accuracy**: **85.24%**  
  - **F1-score**: **81.01%**  

## **7. Results and Discussion**  
Our results confirm that **BERT significantly outperforms SVM**, even with enhancements like **BiLSTM and external embeddings**. **External knowledge augmentation and contrastive learning** improved BERT’s handling of **complex semantic relations**. However, challenges remain, such as **low recall for rare relation types** and **limitations in knowledge source coverage**.

## **8. Use of Generative AI Tools**  
- AI tools such as **ChatGPT** were used for **code debugging, documentation assistance, and conceptual clarifications**.  
- **No AI-generated code was submitted without verification and modification.**  

## **9. Team Members**  
- **Zhuoxuan He**  
- **Kunwei Song**  
- **Xingjian Yuan**  

## **10. References**  
- **SemEval-2010 Task 8 Dataset**: [Dataset Link](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8)  


