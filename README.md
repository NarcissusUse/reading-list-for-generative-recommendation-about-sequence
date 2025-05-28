# 📖 A Reading List for Generative Recommendation

---

## 🎁 Overview

This reading list aims to provide a comprehensive resource guide for researchers and practitioners interested in the field of **Generative Recommendation**. It covers core research directions in recent years, including approaches based on **Large Language Models (LLMs)**, **Semantic IDs**, and **Diffusion Models**.

---

## 📚 Papers

### 📝 Surveys

* **A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys).** KDD 2024. [[paper](https://arxiv.org/abs/2404.00579)]
* **A Survey on LLM-based News Recommender Systems.** Rongyao Wang 2025. [[paper]](https://arxiv.org/abs/2502.09797)
* **Diffusion Models in Recommendation Systems: A Survey.** Ting-Ruen Wei. [[paper]](https://arxiv.org/pdf/2501.10548)
* **A Survey on Sequential Recommendation.** Liwei Pan. [[paper](https://arxiv.org/abs/2412.12770)] 💡

---

### 🗣️ LLM-based Generative Recommendation

#### LLM as Sequential Recommender

##### Early Efforts: Zero-shot Recommendation with LLMs

* **(LLMRank) Large language models are zero-shot rankers for recommender systems.** ECIR 2024. [[paper](https://arxiv.org/pdf/2305.08845)] [[code](https://github.com/RUCAIBox/LLMRank)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LLMRank)
* **(Chat-REC) Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.** arXiv:2303.14524. [[paper](https://arxiv.org/pdf/2303.14524)]
* **(NIR) Zero-Shot Next-Item Recommendation using Large Pretrained Language Models.** arXiv:2304.03153. [[paper](https://arxiv.org/pdf/2304.03153)] [[code](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/AGI-Edgerunners/LLM-Next-Item-Rec)
* **(ChatNews) A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News.** arXiv:2306.10702. [[paper](https://arxiv.org/pdf/2306.10702)] [[code](https://github.com/imrecommender/ChatGPT-News)] ![GitHub Repo stars](https://img.shields.io/github/stars/imrecommender/ChatGPT-News)
* **The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendations.** arXiv:2308.02053. [[paper](https://arxiv.org/pdf/2308.02053)] [[code](https://github.com/Abel2Code/Unequal-Opportunities-of-LLMs)] ![GitHub Repo stars](https://img.shields.io/github/stars/Abel2Code/Unequal-Opportunities-of-LLMs)
* **Is ChatGPT a Good Recommender? A Preliminary Study.** CIKM 2023. [[paper](https://arxiv.org/pdf/2304.10149)] [[code](https://github.com/williamliujl/LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/williamliujl/LLMRec)
* **Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.07609)] [[code](https://github.com/jizhi-zhang/FaiRLLM)] ![GitHub Repo stars](https://img.shields.io/github/stars/jizhi-zhang/FaiRLLM)
* **Uncovering ChatGPT's Capabilities in Recommender Systems.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.02182)] [[code](https://github.com/rainym00d/LLM4RS)] ![GitHub Repo stars](https://img.shields.io/github/stars/rainym00d/LLM4RS)
* **Leveraging Large Language Models for Sequential Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2309.09261)] [[code](https://github.com/dh-r/LLM-Sequential-Recommendation)] ![GitHub Repo stars](https://img.shields.io/github/stars/dh-r/LLM-Sequential-Recommendation)
* **(Rec-GPT4V) Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models.** arXiv:2402.08670. [[paper](https://arxiv.org/pdf/2402.08670)]

##### Aligning LLMs for Recommendation

* **(TallRec) TallRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.00447)] [[code](https://github.com/SAI990323/TALLRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/TALLRec)
* **(GPT4Rec) GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation.** arXiv:2304.03879. [[paper](https://arxiv.org/pdf/2304.03879)]
* **(M6-Rec) M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems.** arXiv:2205.08084. [[paper](https://arxiv.org/pdf/2205.08084)]
* **(BIGRec) A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems.** arXiv:2308.08434. [[paper](https://arxiv.org/pdf/2308.08434)] [[code](https://github.com/SAI990323/BIGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/BIGRec)
* **(InstructRec) Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708882)]
* **(P5) Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/pdf/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)
* **(VIP5) Towards Multimodal Foundation Models for Recommendation.** arXiv:2305.14302. [[paper](https://arxiv.org/pdf/2305.14302)] [[code](https://github.com/jeykigung/VIP5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/VIP5)
* **(GenRec) Generative Recommendation: Towards Next-generation Recommender Paradigm.** arXiv:2304.03516. [[paper](https://arxiv.org/pdf/2304.03516)] [[code](https://github.com/Linxyhaha/GeneRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/GeneRec)
* **(P5-ID) How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3624918.3625339)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)
* **(HKFR) Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM.** RecSys 2023. [[paper](https://arxiv.org/pdf/2308.03333)]
* **(LlamaRec) LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking.** PGAI@CIKM 2023. [[paper](https://arxiv.org/pdf/2311.02089)] [[code](https://github.com/Yueeeeeeee/LlamaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Yueeeeeeee/LlamaRec)
* **(ReLLa) ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation.** arXiv:2308.11131. [[paper](https://arxiv.org/pdf/2308.11131)] [[code](https://github.com/LaVieEnRose365/ReLLa)] ![GitHub Repo stars](https://img.shields.io/github/stars/LaVieEnRose365/ReLLa)
* **(DEALRec) Data-efficient Fine-tuning for LLM-based Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2401.17197)] [[code](https://github.com/Linxyhaha/DEALRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/DEALRec)
* **(CLLM4Rec) Collaborative Large Language Model for Recommender Systems.** WWW 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645347)] [[code](https://github.com/yaochenzhu/llm4rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/yaochenzhu/llm4rec)
* **(RecInterpreter) Large Language Model Can Interpret Latent Space of Sequential Recommender.** arXiv:2310.20487. [[paper](https://arxiv.org/pdf/2310.20487)] [[code](https://github.com/YangZhengyi98/RecInterpreter)] ![GitHub Repo stars](https://img.shields.io/github/stars/YangZhengyi98/RecInterpreter)
* **(TransRec) Bridging Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation.** KDD 2024. [[paper](https://arxiv.org/pdf/2310.06491)] [[code](https://github.com/Linxyhaha/TransRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/TransRec)
* **(RecExplainer) RecExplainer: Aligning Large Language Models for Explaining Recommendation Models.** KDD 2024. [[paper](https://arxiv.org/pdf/2311.10947)] [[code](https://github.com/microsoft/RecAI)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RecAI)
* **(LC-Rec) Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://arxiv.org/pdf/2311.09049)] [[code](https://github.com/RUCAIBox/LC-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LC-Rec)
* **(Collm) Collm: Integrating collaborative embeddings into large language models for recommendation.** arXiv preprint arXiv:2310.19488. [[paper](https://arxiv.org/pdf/2310.19488)] [[code](https://github.com/zyang1580/CoLLM)] ![GitHub Repo stars](https://img.shields.io/github/stars/zyang1580/CoLLM)
* **(E4SRec) E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation.** arXiv:2312.02443. [[paper](https://arxiv.org/pdf/2312.02443)] [[code](https://github.com/HestiaSky/E4SRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/HestiaSky/E4SRec)
* **(Recformer) Text Is All You Need: Learning Language Representations for Sequential Recommendation.** KDD 2023. [[paper](https://arxiv.org/pdf/2305.13731)]
* **(GenRec) GenRec: Large Language Model for Generative Recommendation.** ECIR 2024. [[paper](https://openreview.net/pdf?id=KiX8CW0bCr)] [[code](https://github.com/rutgerswiselab/GenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rutgerswiselab/GenRec)
* **(ONCE) ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models.** WSDM 2024. [[paper](https://arxiv.org/pdf/2305.06566)] [[code](https://github.com/Jyonn/ONCE)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jyonn/ONCE)
* **(ToolRec) Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2405.15114)]
* **(LLaRA) LLaRA: Large Language-Recommendation Assistant.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2312.02445)] [[code](https://github.com/ljy0ustc/LLaRA)] ![GitHub Repo stars](https://img.shields.io/github/stars/ljy0ustc/LLaRA)
* **(I-LLMRec) Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems.** arXiv:2503.06238. [[paper](https://arxiv.org/pdf/2503.06238)] [[code](https://github.com/rlqja1107/torch-I-LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rlqja1107/torch-I-LLMRec)
* **(RALLRec+) RALLRec+: Retrieval Augmented Large Language Model Recommendation with Reasoning.** arXiv:2503.20430. [[paper](https://arxiv.org/pdf/2503.20430)] [[code](https://github.com/sichunluo/RALLRec_plus)] ![GitHub Repo stars](https://img.shields.io/github/stars/sichunluo/RALLRec_plus)

##### Training Objectives & Inference

* **(S-DPO) On Softmax Direct Preference Optimization for Recommendation.** NeurIPS 2024. [[paper](https://arxiv.org/pdf/2406.09215)] [[code](https://github.com/chenyuxin1999/S-DPO)] ![GitHub Repo stars](https://img.shields.io/github/stars/chenyuxin1999/S-DPO)
* **(D3) Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation.** EMNLP 2024. [[paper](https://arxiv.org/pdf/2406.14900)] [[code](https://github.com/SAI990323/DecodingMatters)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/DecodingMatters)
* **(SLMREC) SLMREC: Empowering Small Language Models for Sequential Recommendation.** arXiv:2405.17890. [[paper](https://arxiv.org/pdf/2405.17890)] [[code](https://github.com/WujiangXu/SLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/WujiangXu/SLMRec)

#### LLM as Conversational Recommender & Recommendation Assistant

* **(LLM-REDIAL) LLM-REDIAL: A Large-Scale Dataset for Conversational Recommender Systems Created from User Behaviors with LLMs.** ACL Findings 2024. [[paper](https://aclanthology.org/2024.findings-acl.529.pdf)] [[code](https://github.com/LitGreenhand/LLM-Redial)] ![GitHub Repo stars](https://img.shields.io/github/stars/LitGreenhand/LLM-Redial)
* **(iEvaLM) Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models.** EMNLP 2023. [[paper](https://arxiv.org/pdf/2305.13112)] [[code](https://github.com/RUCAIBox/iEvaLM-CRS)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/iEvaLM-CRS)
* **How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation.** WWW 2024. [[paper](https://arxiv.org/pdf/2403.16416)] [[code](https://github.com/RUCAIBox/iEvaLM-CRS)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/iEvaLM-CRS)
* **Large Language Models as Zero-Shot Conversational Recommenders.** CIKM 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3614949)] [[code](https://github.com/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys)] ![GitHub Repo stars](https://img.shields.io/github/stars/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys)
* **(InteRecAgent) Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations.** arXiv:2308.16505. [[paper](https://arxiv.org/pdf/2308.16505)] [[code](https://github.com/microsoft/RecAI)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RecAI)
* **(RecMind) RecMind: Large Language Model Powered Agent For Recommendation.** NACCL 2024 (Findings). [[paper](https://arxiv.org/pdf/2308.14296)]
* **(RAH) RAH! RecSys–Assistant–Human: A Human-Centered Recommendation Framework With LLM Agents.** TOCS 2024. [[paper](https://arxiv.org/pdf/2308.09904)]
* **(MACRec) MACRec: A Multi-Agent Collaboration Framework for Recommendation.** SIGIR 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3626772.3657669)] [[code](https://github.com/wzf2000/MACRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/wzf2000/MACRec)

#### LLM as User Simulator

* **(RecAgent) User Behavior Simulation with Large Language Model-based Agents for Recommender Systems.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708985)] [[code](https://github.com/RUC-GSAI/YuLan-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUC-GSAI/YuLan-Rec)
* **(AgentCF) AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems.** WWW 2024. [[paper](https://arxiv.org/pdf/2310.09233)]
* **(Agent4Rec) On Generative Agents in Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2310.10108)] [[code](https://github.com/LehengTHU/Agent4Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/LehengTHU/Agent4Rec)
* **LLM-Powered User Simulator for Recommender System.** arXiv:2412.16984. [[paper](https://arxiv.org/pdf/2412.16984)]
* **Enhancing Cross-Domain Recommendations with Memory-Optimized LLM-Based User Agents.** arXiv:2502.13843. [[paper](https://arxiv.org/pdf/2502.13843)]
* **FLOW: A Feedback LOop FrameWork for Simultaneously Enhancing Recommendation and User Agents.** arXiv:2410.20027. [[paper](https://arxiv.org/pdf/2410.20027)]
* **A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems.** arXiv:2405.08035. [[paper](https://arxiv.org/pdf/2405.08035)] [[code](https://github.com/zlxxlz1026/CSHI)] ![GitHub Repo stars](https://img.shields.io/github/stars/zlxxlz1026/CSHI)

---

### 🆔 Semantic ID-based Generative Recommendation

#### Semantic ID Construction

##### Quantization

* **(VQ-Rec) Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders.** WWW 2023. [[paper](https://arxiv.org/abs/2210.12316)] [[code](https://github.com/RUCAIBox/VQ-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/VQ-Rec)
* **(TIGER) Recommender Systems with Generative Retrieval.** NeurIPS 2023. [[paper](https://arxiv.org/abs/2305.05065)]
* **Generative Sequential Recommendation with GPTRec.** Gen-IR @ SIGIR 2023 workshop. [[paper](https://arxiv.org/abs/2306.11114)]
* **(ColaRec) Content-Based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/abs/2403.18480)] [[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)
* **CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation.** RecSys 2024. [[paper](https://arxiv.org/abs/2404.14774)]
* **MMGRec: Multimodal Generative Recommendation with Transformer Model.** arXiv:2404.16555. [[paper](https://arxiv.org/abs/2404.16555)]
* **(LETTER) Learnable Item Tokenization for Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.07314)] [[code](https://github.com/HonghuiBao2000/LETTER)] ![GitHub Repo stars](https://img.shields.io/github/stars/HonghuiBao2000/LETTER)
* **(ETEGRec) End-to-End Learnable Item Tokenization for Generative Recommendation.** arXiv:2409.05546. [[paper](https://arxiv.org/abs/2409.05546)]
* **(MoC) Towards Scalable Semantic Representation for Recommendation.** arXiv:2410.09560. [[paper](https://arxiv.org/abs/2410.09560)]

##### Hierarchical Clustering

* **(DSI) Transformer Memory as a Differentiable Search Index.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2202.06991)]
* **(NCI) A Neural Corpus Indexer for Document Retrieval.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2206.02743)] [[code](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)] ![GitHub Repo stars](https://img.shields.io/github/stars/solidsea98/Neural-Corpus-Indexer-NCI)
* **How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://arxiv.org/abs/2305.06569)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)
* **(SEATER) Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning.** SIGIR-AP 2024. [[paper](https://arxiv.org/abs/2309.13375)] [[code](https://github.com/ethan00si/seater_generative_retrieval)] ![GitHub Repo stars](https://img.shields.io/github/stars/ethan00si/seater_generative_retrieval)

##### Contextual Action Tokenization

* **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation.** arXiv:2502.13581. [[paper](https://arxiv.org/abs/2502.13581)]

##### Behavior-aware Tokenization

* **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration.** KDD 2024. [[paper](https://arxiv.org/abs/2406.14017)] [[code](https://github.com/yewzz/EAGER)] ![GitHub Repo stars](https://img.shields.io/github/stars/yewzz/EAGER)
* **SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for Sequential Recommendation.** arXiv:2408.08686. [[paper](https://arxiv.org/abs/2408.08686)]
* **(MBGen) Multi-Behavior Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.16871)] [[code](https://github.com/anananan116/MBGen)] ![GitHub Repo stars](https://img.shields.io/github/stars/anananan116/MBGen)
* **(PRORec) Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation.** arXiv:2502.06269. [[paper](https://arxiv.org/abs/2502.06269)]

##### Language Model-based Generator

* **(LMIndexer) Language Models As Semantic Indexers.** ICDE 2024. [[paper](https://arxiv.org/abs/2310.07815)] [[code](https://github.com/PeterGriffinJin/LMIndexer)] ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/LMIndexer)
* **IDGenRec: LLM-RecSys Alignment with Textual ID Learning.** SIGIR 2024. [[paper](https://arxiv.org/abs/2403.19021)] [[code](https://github.com/agiresearch/IDGenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/IDGenRec)

#### Architecture

##### Dense & Generative Retrieval

* **(SpecGR) Inductive Generative Recommendation via Retrieval-based Speculation.** arXiv:2410.02939. [[paper](https://arxiv.org/abs/2410.02939)] [[code](https://github.com/Jamesding000/SpecGR)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jamesding000/SpecGR)
* **(LIGER) Unifying Generative and Dense Retrieval for Sequential Recommendation.** arXiv:2411.18814. [[paper](https://arxiv.org/abs/2411.18814)]

##### Unified Retrieval and Ranking

* **(HSTU) Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.** ICML 2024. [[paper](https://arxiv.org/abs/2402.17152)] [[code](https://github.com/facebookresearch/generative-recommenders)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/generative-recommenders)
* **OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment.** arXiv:2502.18965. [[paper](https://arxiv.org/abs/2502.18965)]

#### Aligning with LLMs

* **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/abs/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)
* **(LC-Rec) Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://arxiv.org/abs/2311.09049)] [[code](https://github.com/RUCAIBox/LC-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LC-Rec)
* **(AtSpeed) Efficient Inference for Large Language Model-based Generative Recommendation.** ICLR 2025. [[paper](https://arxiv.org/abs/2410.05165)] [[code](https://github.com/Linxyhaha/AtSpeed)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/AtSpeed)
* **Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization.** AAAI 2025. [[paper](https://arxiv.org/abs/2412.13771)]
* **EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration.** WWW 2025. [[paper](https://arxiv.org/abs/2502.14735)]

---

### 🎨 Diffusion Model-based Generative Recommendation

#### Diffusion-enhanced Recommendation

* **Diffusion Augmentation for Sequential Recommendation.** CIKM 2023. [[paper](https://arxiv.org/abs/2309.12858)]
* **Diff4Rec: Sequential Recommendation with Curriculum-scheduled Diffusion Augmentation.** MM 2023. [[paper](https://dl.acm.org/doi/10.1145/3581783.3612709)]
* **DiffMM: Multi-Modal Diffusion Model for Recommendation.** MM 2024. [[paper](https://arxiv.org/abs/2406.11781)]
* **Conditional Denoising Diffusion for Sequential Recommendation.** PAKDD 2024. [[paper](https://arxiv.org/abs/2304.11433)]
* **Diffkg: Knowledge Graph Diffusion Model for Recommendation.** WSDM 2024. [[paper](https://arxiv.org/abs/2312.16890)]

#### Diffusion as Recommender

* **Diffusion Recommender Model.** SIGIR 2023. [[paper](https://arxiv.org/abs/2304.04971)]
* **Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion.** NeurIPS 2023. [[paper](https://arxiv.org/abs/2310.20453)]
* **Bridging User Dynamics: Transforming Sequential Recommendations with Schrödinger Bridge and Diffusion Models.** CIKM 2024. [[paper](https://arxiv.org/abs/2409.10522)]
* **DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models.** CIKM 2024. [[paper](https://arxiv.org/abs/2408.12153)]
* **Plug-in Diffusion Model for Sequential Recommendation.** AAAI 2024. [[paper](https://arxiv.org/abs/2401.02913)]
* **SeeDRec: Sememe-based Diffusion for Sequential Recommendation.** IJCAI 2024. [[paper](https://www.ijcai.org/proceedings/2024/251)]
* **Breaking Determinism: Fuzzy Modeling of Sequential Recommendation Using Discrete State Space Diffusion Model.** NeurIPS 2024. [[paper](https://arxiv.org/abs/2410.23994)]
* **Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation.** WWW 2025. [[paper](https://arxiv.org/abs/2501.17670)]
* **Preference Diffusion for Recommendation.** ICLR 2025. [[paper](https://arxiv.org/abs/2410.13117)]
* **Recommendation via Collaborative Diffusion Generative Model.** KSEM 2022. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-10989-8_47)]
* **G-Diff: A Graph-Based Decoding Network for Diffusion Recommender Model.** IEEE TNNLS 2024. [[paper](https://ieeexplore.ieee.org/document/10750895)]
* **Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity.** SIGIR 2024. [[paper](https://arxiv.org/abs/2404.14240)] [[code](https://github.com/jackfrost168/CF_Diff)] ![GitHub Repo stars](https://img.shields.io/github/stars/jackfrost168/CF_Diff)
* **Denoising Diffusion Recommender Model.** SIGIR 2024. [[paper](https://arxiv.org/abs/2401.06982)] [[code](https://github.com/Polaris-JZ/DDRM)] ![GitHub Repo stars](https://img.shields.io/github/stars/Polaris-JZ/DDRM)
* **DGRM: Diffusion-GAN Recommendation Model to Alleviate the Mode Collapse Problem in Sparse Environments.** Pattern Recognition 2024. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324004436)]
* **Stochastic Sampling for Contrastive Views and Hard Negative Samples in Graph-based Collaborative Filtering.** WSDM 2025. [[paper](https://arxiv.org/abs/2405.00287)] [[code](https://github.com/jeongwhanchoi/SCONE)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeongwhanchoi/SCONE)
* **RecDiff: Diffusion Model for Social Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2406.01629)] [[code](https://github.com/HKUDS/RecDiff)] ![GitHub Repo stars](https://img.shields.io/github/stars/HKUDS/RecDiff)
* **Blurring-Sharpening Process Models for Collaborative Filtering.** SIGIR 2023. [[paper](https://arxiv.org/abs/2211.09324)]
* **Graph Signal Diffusion Model for Collaborative Filtering.** SIGIR 2024. [[paper](https://arxiv.org/abs/2311.08744)]
* **Diffurec: A Diffusion Model for Sequential Recommendation.** TOIS 2023. [[paper](https://arxiv.org/abs/2304.00686)]
* **A Diffusion Model for POI Recommendation.** TOIS 2023. [[paper](https://arxiv.org/abs/2304.07041)]
* **Towards Personalized Sequential Recommendation via Guided Diffusion.** ICIC 2024. [[paper](https://dl.acm.org/doi/10.1007/978-981-97-5618-6_1)]
* **Diffusion Recommendation with Implicit Sequence Influence.** WebConf 2024. [[paper](https://dl.acm.org/doi/10.1145/3589335.3651951)]
* **Uncertainty-aware Guided Diffusion for Missing Data in Sequential Recommendation.** SIGIR 2025. [[paper](https://openreview.net/forum?id=w2HL7yuWE2)]
* **Generate and Instantiate What You Prefer: Text-Guided Diffusion for Sequential Recommendation.** arXiv 2024. [[paper](https://arxiv.org/abs/2410.13428)]

#### Personalized Content Generation with Diffusion

* **DreamVTON: Customizing 3D Virtual Try-on with Personalized Diffusion Models.** MM2024. [[paper](https://arxiv.org/abs/2407.16511)]
* **Instant 3D Human Avatar Generation using Image Diffusion Models.** ECCV 2024. [[paper](https://arxiv.org/abs/2406.07516)]
* **Subject-Diffusion: Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning.** SIGGRAPH 2024. [[paper](https://arxiv.org/abs/2307.11410)] [[code](https://github.com/OPPO-Mente-Lab/Subject-Diffusion)] ![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-Mente-Lab/Subject-Diffusion)
* **Diffusion Models for Generative Outfit Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/abs/2402.17279)]
