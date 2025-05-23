
# 🦅 HawkBench: A Benchmark for Stratified Information-Seeking with RAG

📄 **Paper:** [HAWKBENCH: Investigating Resilience of RAG Methods on Stratified Information-Seeking Tasks](https://arxiv.org/pdf/2502.13465)  
🧠 **Authors:** Hongjin Qian, Zheng Liu, Chao Gao, Yankai Wang, Defu Lian, Zhicheng Dou  
📦 **Dataset:** [HuggingFace](https://huggingface.co/datasets/TommyChien/HawkBench) (1600 annotated QA samples across 8 domains)

---

## 🧠 What is HawkBench?

HawkBench is a **human-labeled, multi-domain benchmark** designed to evaluate the *resilience* of Retrieval-Augmented Generation (RAG) systems across a **stratified range of information-seeking tasks**. Unlike most RAG benchmarks that focus on isolated settings (e.g., factoid QA or legal queries), HawkBench introduces:

- **Four Task Levels**: Based on *Referencing* and *Reasoning* complexity:
  - 🔹 **Level 1**: Explicit Factoid
  - 🔹 **Level 2**: Implicit Factoid
  - 🔸 **Level 3**: Explicit Rationale
  - 🔸 **Level 4**: Implicit Rationale
- **8 Diverse Domains**: Technology, Novel, Art, Humanities, Paper, Science, Finance, Legal
- **Rigorous Annotation**: Generated by GPT-4o and DeepSeek-V3, refined by human PhD annotators

---

## 🚀 Getting Started

### 🔎 Annotation Demo

```bash
streamlit run annotation/demo.py

🔁 Retrieval App

python annotation/retrieval_app.py

🧹 Post-Process Annotated Data

python annotation/process.py
```



## 📊 Evaluation

Run Evaluation Pipeline
```bash
bash scripts/eval.sh
```



## 📦 Dataset

Download and extract:
```bash
tar -xvf data/HawkBench.tar.gz
```
or use:
👉 [HuggingFace](https://huggingface.co/datasets/TommyChien/HawkBench)



## 🧪 Experiments


📈 Main Results

| Method | Level 1 F1 | Level 2 F1 | Level 3 S-F1 | Level 4 S-F1 |
|--------|------------|------------|--------------|--------------|
| LLM | 12.9 | 11.5 | 24.0 | 33.2 |
| Lingua-2 | 11.4 | 11.4 | 23.9 | 25.2 |
| MInference | 11.1 | 11.2 | 24.2 | 33.3 |
| RAG | 57.5 | 38.6 | 27.3 | 18.3 |
| HyDE | 73.5 | 44.5 | 28.0 | 18.4 |
| RQRAG | 73.6 | 46.8 | 28.6 | 17.4 |
| MemoRAG | 50.2 | 37.3 | 34.1 | 35.0 |
| GraphRAG | 57.4 | 37.0 | 32.5 | 28.7 |





## 📚 Citation
```bibtex
@article{qian2024hawkbench,
  title={HAWKBENCH: Investigating Resilience of RAG Methods on Stratified Information-Seeking Tasks},
  author={Hongjin Qian and Zheng Liu and Chao Gao and Yankai Wang and Defu Lian and Zhicheng Dou},
  journal={arXiv preprint arXiv:2502.13465},
  year={2024}
}

