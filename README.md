<div align="center">
<h2>    
A Reading List for Generative recommandation 


## Table of Contents
- [🎁 Surveys](#gift-surveys)
- [🚀 LLM as SequentialRecommender ]
  - [Early efforts: pretrained LLMs for rec]
  - [Aligning LLMs for recommendation]
  - [Training objective & inference]
  - [Efficiency](#pddl+local-search)
- [🧠 Planning](#brain-planning)
  - [Base Workflows](#base-workflows)
  - [Search Workflows](#search-workflows)
  - [Decomposition](#decomposition)
  - [PDDL + Local Search](#pddl+local-search)
  - [Others](#others)
- [🔄 Feedback Learning](#arrows_counterclockwise-feedback-learning)
- [🧩 Composition](#jigsaw-composition)
  - [Planning + Feedback Learning](#planning--feedback-learning)
  - [Planning + Tool Use](#planning--tool-use)
- [🌍 World Modeling](#world_map-world-modeling)
  <!-- - [LLM as World Models](#llm-as-world-models)
  - [LLM-Generated World Models](#llm-generated-world-models) -->
- [📊 Benchmarks](#bar_chart-benchmarks)
- [📝 Citation](#memo-citation)


## :gift: Surveys
- **A Survey on LLM-based News Recommender Systems**, Rongyao Wang 2025 [[paper]](https://arxiv.org/abs/2502.09797))
- **Diffusion Models in Recommendation Systems: A Survey**, Ting-Ruen Wei [[paper]](https://arxiv.org/pdf/2501.10548) 
- **A Survey on Sequential Recommendation**, Liwei Pan [[paper]](https://arxiv.org/abs/2412.12770) 💡
- **A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys).** KDD 2024. [[paper](https://arxiv.org/abs/2404.00579)]

## use LLM as recommender 
### :rocket: LLM as SequentialRecommender
* (LLMRank) **Large language models are zero-shot rankers for recommender systems.** ECIR 2024. [[paper](https://arxiv.org/pdf/2305.08845)] [[code](https://github.com/RUCAIBox/LLMRank)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LLMRank)

   *Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, Wayne Xin Zhao.*

* (Chat-REC) **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.** arXiv:2303.14524. [[paper](https://arxiv.org/pdf/2303.14524)] 

   *Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, Jiawei Zhang.*

* (NIR) **Zero-Shot Next-Item Recommendation using Large Pretrained Language Models.** arXiv:2304.03153. [[paper](https://arxiv.org/pdf/2304.03153)] [[code](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/AGI-Edgerunners/LLM-Next-Item-Rec)

   *Lei Wang, Ee-Peng Lim.*

* (ChatNews) **A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News.** arXiv:2306.10702. [[paper](https://arxiv.org/pdf/2306.10702)] [[code](https://github.com/imrecommender/ChatGPT-News)] ![GitHub Repo stars](https://img.shields.io/github/stars/imrecommender/ChatGPT-News)

   *Xinyi Li, Yongfeng Zhang, Edward C. Malthouse.*

* **The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendations.** arXiv:2308.02053. [[paper](https://arxiv.org/pdf/2308.02053)] [[code](https://github.com/Abel2Code/Unequal-Opportunities-of-LLMs)] ![GitHub Repo stars](https://img.shields.io/github/stars/Abel2Code/Unequal-Opportunities-of-LLMs)

   *Abel Salinas, Parth Vipul Shah, Yuzhong Huang, Robert McCormack, Fred Morstatter.*

* **Is ChatGPT a Good Recommender? A Preliminary Study.** CIKM 2023. [[paper](https://arxiv.org/pdf/2304.10149)] [[code](https://github.com/williamliujl/LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/williamliujl/LLMRec)

   *Junling Liu, Chao Liu, Peilin Zhou, Renjie Lv, Kang Zhou, Yan Zhang.*

* **Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.07609)] [[code](https://github.com/jizhi-zhang/FaiRLLM)] ![GitHub Repo stars](https://img.shields.io/github/stars/jizhi-zhang/FaiRLLM)

   *Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He.*

* **Uncovering ChatGPT's Capabilities in Recommender Systems.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.02182)] [[code](https://github.com/rainym00d/LLM4RS)] ![GitHub Repo stars](https://img.shields.io/github/stars/rainym00d/LLM4RS)

   *Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, Jun Xu.*

* **Leveraging Large Language Models for Sequential Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2309.09261)] [[code](https://github.com/dh-r/LLM-Sequential-Recommendation)] ![GitHub Repo stars](https://img.shields.io/github/stars/dh-r/LLM-Sequential-Recommendation)

   *Jesse Harte, Wouter Zorgdrager, Panos Louridas, Asterios Katsifodimos, Dietmar Jannach, Marios Fragkoulis.*

* (Rec-GPT4V) **Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models.** arXiv:2402.08670. [[paper](https://arxiv.org/pdf/2402.08670)]

   *Yuqing Liu, Yu Wang, Lichao Sun, Philip S. Yu.*


* (LLaRA) **LLaRA: Large Language-Recommendation Assistant.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2312.02445)] [[code](https://github.com/ljy0ustc/LLaRA)] ![GitHub Repo stars](https://img.shields.io/github/stars/ljy0ustc/LLaRA)

   *Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, Xiangnan He.*

* (I-LLMRec) **Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems.** arXiv:2503.06238. [[paper](https://arxiv.org/pdf/2503.06238)] [[code](https://github.com/rlqja1107/torch-I-LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rlqja1107/torch-I-LLMRec)

   *Kibum Kim, Sein Kim, Hongseok Kang, Jiwan Kim, Heewoong Noh, Yeonjun In, Kanghoon Yoon, Jinoh Oh, Chanyoung Park.*

* (RALLRec+) **RALLRec+: Retrieval Augmented Large Language Model Recommendation with Reasoning.** arXiv:2503.20430. [[paper](https://arxiv.org/pdf/2503.20430)] [[code](https://github.com/sichunluo/RALLRec_plus)] ![GitHub Repo stars](https://img.shields.io/github/stars/sichunluo/RALLRec_plus)

   *Sichun Luo, Jian Xu, Xiaojie Zhang, Linrong Wang, Sicong Liu, Hanxu Hou, Linqi Song.*

### LLM as ConversationalRecommende



### LLM as User Simulator
- **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**, NeurIPS 2023 [[paper](https://openreview.net/forum?id=yHdTscY6Ci)] | [[code]](https://github.com/microsoft/JARVIS/tree/main/hugginggpt)
- **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**, NeurIPS 2023 [[paper](https://openreview.net/forum?id=WZH7099tgfM)]
## Semantic ID
### Semantic ID Construction
- **Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning**, NeurIPS 2023 [[paper](https://arxiv.org/abs/2305.14909)] | [[code]](https://github.com/GuanSuns/LLMs-World-Models-for-Planning)
- **On the Planning Abilities of Large Language Models - A Critical Investigation**, NeurIPS 2023 [[paper](https://openreview.net/forum?id=X6dEqXIsEW)] | [[code]](https://github.com/karthikv792/LLMs-Planning)
- **PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change**, NeurIPS 2023 [[paper](https://openreview.net/forum?id=YXogl4uQUO)] | [[code]](https://github.com/karthikv792/LLMs-Planning)

### :SemID-based Generative Recommendation Model Architecture
- **LLM+P: Empowering Large Language Models with Optimal Planning Proficiency**, arXiv.2304.11477 [[paper](https://doi.org/10.48550/arXiv.2304.11477)]💡
## Use diffusion in Reccommender
### :Diffusion as enhancer
- **Reflexion: Language Agents with Verbal Reinforcement Learning**, NeurIPS 2023 [[paper](https://doi.org/10.48550/arXiv.2303.11366)]
- **Self-Refine: Iterative Refinement with Self-Feedback**, NeurIPS 2023 [[paper](https://doi.org/10.48550/arXiv.2303.17651)]
- **SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning**, ICLR 2024 [[paper]](https://arxiv.org/abs/2308.00436) | [[code]](https://github.com/ningmiao/selfcheck)
- **Learning From Correctness Without Prompting Makes LLM Efficient Reasoner**, COLM2024 [[paper]](https://openreview.net/forum?id=dcbNzhVVQj#discussion)
- **Learning From Mistakes Makes LLM Better Reasoner**, arXiv [[paper]](https://arxiv.org/abs/2310.20689) | [[code]](https://github.com/microsoft/LEMA)💡
- **LLM-based Rewriting of Inappropriate Argumentation using Reinforcement Learning from Machine Feedback** ACL 2024 [[paper]](https://arxiv.org/abs/2406.03363)


### :Diffusion as recommender

### :bar_chart: Benchmarks
#### Tool-Use Benchmarks
- **MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use**, arXiv.2310.03128 [[paper](https://doi.org/10.48550/arXiv.2310.03128)] 💡
- **TaskBench: Benchmarking Large Language Models for Task Automation**, arXiv.2311.18760 [[paper](https://doi.org/10.48550/arXiv.2311.18760)] 💡

#### Planning Benchmarks
- **Large Language Models Still Can't Plan (A Benchmark for LLMs on Planning and Reasoning about Change)**, NeurIPS 2023 [[paper](https://doi.org/10.48550/arXiv.2206.10498)] 


