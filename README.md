# Dissertation Title: Financial Event Extraction and Classification using NLP and LLMs

## Overview

This project focuses on the development of a robust system for the extraction and classification of financial events from unstructured textual data using advanced Natural Language Processing (NLP) techniques and Large Language Models (LLMs). The primary goal is to enhance the accuracy and timeliness of financial event detection, such as corporate announcements and significant market-moving news, which can be crucial for financial analysts, automated trading systems, and decision-makers.

## Project Objectives

1. **Zero-Shot Learning for Event Extraction**: The first objective is to apply zero-shot learning techniques using LLMs such as FlanT5, GPT-2, and BART to evaluate their performance in financial event extraction without requiring any prior task-specific training. This approach helps assess the models' inherent understanding and ability to generalize to new tasks.

2. **Few-Shot Learning Approaches**: Building on the insights from zero-shot learning, this objective involves employing few-shot learning strategies. Specifically, couple of selection strategies are used to pick examples from each event category, enabling the models to refine their event classification capabilities and improve accuracy with minimal labeled data.

3. **Fine-Tuning LLMs**: This objective focuses on fine-tuning LLMs by training them on the "Trade the Event" dataset. The trained models are evaluated on the development/test dataset to assess their performance using relevant metrics. Further validation is conducted by running the models on individual financial articles to test their predictive capabilities.

4. **Traditional Machine Learning Classification**: As a complementary approach to LLM-based methods, this objective explores traditional machine learning classification techniques. Various Machine Learning classifiers are employed to identify the best-performing algorithm for event classification. Additionally, a simple Long Short-Term Memory (LSTM) architecture is developed to further evaluate classification performance.

## Methodology Overview

The methodology adopted in this project involves a combination of state-of-the-art LLMs and traditional machine learning techniques. The key steps include:

### 1. **Data Collection and Preprocessing**
   - Utilize the "Trade the Event" dataset, which is rich with financial articles annotated with labels corresponding to 11 types of corporate events.
   - Perform data cleaning and preprocessing to ensure the text data is in a suitable format for model training and evaluation.

### 2. **Zero-Shot and Few-Shot Learning**
   - Implement zero-shot learning techniques using LLMs to evaluate their performance without task-specific training.
   - Employ few-shot learning strategies with random selection to refine the models' understanding and improve classification accuracy with minimal labeled data.

### 3. **Fine-Tuning LLMs**
   - Fine-tune LLMs such as FlanT5, GPT-2, and BART on the provided dataset to enhance their event extraction and classification capabilities.
   - Evaluate the fine-tuned models on a development/test dataset and validate their performance on individual financial articles.

### 4. **Traditional Machine Learning Classification**
   - Explore traditional machine learning classifiers, such as Logistic Regression, SVM, and Gradient Boosting, to compare their performance against LLM-based methods.
   - Develop and evaluate an LSTM architecture to further assess classification performance.


The results of this project not only contribute to the academic understanding of financial event extraction but also offer practical implications for the development of automated trading systems and decision-making tools in the finance industry.


## References

1. Jean Lee, Nicholas Stevens, Soyeon Caren Han, and Minseok Song. A Survey of Large Language Models in Finance (FinLLMs). Papers 2402.02315, arXiv.org, February 2024. URL: [https://ideas.repec.org/p/arx/papers/2402.02315.html](https://ideas.repec.org/p/arx/papers/2402.02315.html).
2. Jean Lee, Nicholas Stevens, Soyeon Caren Han, and Minseok Song. A Survey of Large Language Models in Finance (FinLLMs), 2024. URL: [https://arxiv.org/abs/2402.02315v1](https://arxiv.org/abs/2402.02315v1).
3. Zhaoyue Sun, Gabriele Pergola, Byron C. Wallace, and Yulan He. Leveraging ChatGPT in Pharmacovigilance Event Extraction: An Empirical Study, 2024. URL: [https://arxiv.org/abs/2402.15663](https://arxiv.org/abs/2402.15663).
4. Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A Survey of Large Language Models, 2023. URL: [https://arxiv.org/abs/2303.18223](https://arxiv.org/abs/2303.18223).
5. Zhihan Zhou, Liqian Ma, and Han Liu. Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2114–2124, Online, August 2021. Association for Computational Linguistics. doi: [10.18653/v1/2021.findings-acl.186](https://aclanthology.org/2021.findings-acl.186). URL: [https://aclanthology.org/2021.findings-acl.186](https://aclanthology.org/2021.findings-acl.186).

