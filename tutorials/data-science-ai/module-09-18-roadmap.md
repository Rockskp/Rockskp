# Modules 9–18 + Capstones — Compact Study Roadmap (outline)
### मॉड्यूल 9–18 + Capstones — संक्षिप्त study roadmap

> **Status: OUTLINE.** Modules 1–8 are full lessons with verified code. Modules 9–18 below are condensed study maps: key concepts in English + Hindi, one "याद रखो" rule, interview traps, a mini-project spec and primary sources. They will be expanded to the full lesson format (code, flashcards, exercises, quizzes) in a follow-up.

---

## Module 9 — Time Series, A/B Testing, Causal ML
- **Concepts:** trend/seasonality/lag; rolling windows, stationarity (ADF); ACF/PACF; exponential smoothing, ARIMA/SARIMA; ML forecasting with lag features + gradient boosting; **walk-forward backtesting** (never shuffle); A/B test design (power, sample size, peeking); confounders, DAGs, propensity scores, difference-in-differences.
- **हिन्दी:** Time series में भविष्य से कभी train मत करो — validation हमेशा आगे के समय पर। A/B test में पहले sample size तय करो, बीच में बार-बार p-value मत देखो। Correlation cause नहीं; confounder को DAG से पहचानो।
- **याद रखो:** *"समय आगे चलता है — split भी आगे।"*
- **Traps:** random CV on time data; leakage through rolling features computed with the current row; stopping an A/B test when p < 0.05.
- **Mini-project:** Demand forecasting on `datasets/flights.csv` — seasonal-naive baseline vs SARIMA vs LightGBM with lags, walk-forward MAE/WAPE.
- **Sources:** Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* https://otexts.com/fpp3/ · statsmodels TSA https://www.statsmodels.org/stable/tsa.html · Kohavi et al., *Trustworthy Online Controlled Experiments* · Hernán & Robins, *Causal Inference: What If* https://miguelhernan.org/whatifbook

## Module 10 — Deep Learning with PyTorch
- **Concepts:** neuron (weights, bias, activation), MLP forward pass, loss (MSE → cross-entropy), **backpropagation = chain rule**, activations (ReLU, GELU, sigmoid), vanishing/exploding gradients, init, regularisation (dropout, weight decay, early stopping), tensors/autograd, `Dataset`/`DataLoader`/`nn.Module`, training loop with validation + checkpoints, 4 GB GPU tactics (AMP, grad accumulation — Module 1.4).
- **हिन्दी:** Neural network = कई layers का function; backprop chain rule से हर weight का gradient निकालता है। Training loop: forward → loss → backward → optimizer.step → zero_grad।
- **याद रखो:** *"zero_grad भूले तो gradients जुड़ते जाएँगे।"*
- **Traps:** forgetting `model.eval()` / `torch.no_grad()` at validation; softmax before `CrossEntropyLoss`.
- **Mini-project:** NumPy 2-layer net from scratch on penguins, then the same in PyTorch; checkpoint best validation loss.
- **Sources:** PyTorch tutorials https://docs.pytorch.org/tutorials/ · Goodfellow et al., *Deep Learning* https://www.deeplearningbook.org/ · Rumelhart, Hinton, Williams 1986 (backprop).

## Module 11 — Computer Vision
- **Concepts:** images as tensors (C×H×W, normalisation), convolution (kernel, padding, stride), pooling, receptive field, augmentation, CNN architectures (LeNet → ResNet → EfficientNet), transfer learning (frozen backbone → fine-tune selected layers), detection (IoU, NMS, YOLO idea), segmentation (U-Net), Vision Transformers, image search with embeddings.
- **हिन्दी:** Convolution छोटे filters से edges/patterns पकड़ता है; transfer learning में pretrained model का backbone रखो, सिर्फ़ ऊपर की layer train करो — कम data, कम GPU।
- **याद रखो:** *"Output size = (W − K + 2P)/S + 1."*
- **Mini-project:** CIFAR-10 / digits classifier: small CNN vs frozen ResNet-18 features; image search engine via cosine similarity on embeddings.
- **Sources:** He et al., *ResNet* 2015 https://arxiv.org/abs/1512.03385 · Dosovitskiy et al., *ViT* 2020 https://arxiv.org/abs/2010.11929 · Ronneberger et al., *U-Net* https://arxiv.org/abs/1505.04597 · torchvision https://docs.pytorch.org/vision/stable/

## Module 12 — NLP and Transformers from Scratch
- **Concepts:** cleaning/tokenisation, stemming/lemmatisation, n-grams, bag-of-words, TF-IDF, Word2Vec, RNN/LSTM/GRU and vanishing gradients, **attention (Q, K, V)**, scaled dot-product softmax(QKᵀ/√d)V, multi-head attention, positional encoding, residuals + LayerNorm, encoder vs decoder, causal masking, tiny char-level GPT.
- **हिन्दी:** Attention हर token को बाकी tokens से "पूछने" देता है कि कौन ज़रूरी है; √d से divide करने पर softmax saturate नहीं होता; causal mask भविष्य के tokens छिपाता है।
- **याद रखो:** *"Query पूछता है, Key पहचान है, Value जवाब है।"*
- **Mini-project:** TF-IDF + logistic baseline, then a ~1M-parameter char-level GPT trained on CPU/Colab.
- **Sources:** Vaswani et al., *Attention Is All You Need* 2017 https://arxiv.org/abs/1706.03762 · Mikolov et al., *Word2Vec* https://arxiv.org/abs/1301.3781 · Karpathy, *nanoGPT* https://github.com/karpathy/nanoGPT

## Module 13 — LLM Engineering, Embeddings and RAG
- **Concepts:** tokens, context windows, chat messages; pretraining → SFT → preference tuning; decoding (temperature, top-k, top-p); running small models locally (Hugging Face Transformers); sentence embeddings + semantic search; vector indexes (flat, IVF, HNSW); **RAG**: chunking, retrieval, context assembly, citations; quality: reranking, hallucination checks, evaluation (faithfulness, answer relevance, recall@k).
- **हिन्दी:** RAG में पहले सही chunks ढूँढो, फिर उन्हें prompt में डालकर citation के साथ जवाब दो। Retrieval ख़राब तो जवाब ख़राब — पहले retrieval नापो।
- **याद रखो:** *"Garbage retrieved, garbage generated."*
- **Traps:** chunks too large/small, no metadata filters, no golden eval set, prompt injection from documents.
- **Mini-project:** Local chat-with-documents RAG over this tutorial's own Markdown (TF-IDF or sentence-transformers + FAISS/NumPy), with cited answers and a 20-question eval set.
- **Sources:** Lewis et al., *RAG* 2020 https://arxiv.org/abs/2005.11401 · Hugging Face docs https://huggingface.co/docs/transformers · Malkov & Yashunin, *HNSW* https://arxiv.org/abs/1603.09320 · Sentence-BERT https://arxiv.org/abs/1908.10084

## Module 14 — Efficient Fine-Tuning and Alignment
- **Concepts:** pretraining vs fine-tuning vs instruction tuning; PEFT/adapters; **LoRA**: W + (α/r)·BA with low rank r; quantization INT8/INT4/NF4; **QLoRA** (4-bit frozen base + LoRA adapters); alignment: RLHF, reward models, DPO.
- **हिन्दी:** LoRA पूरे weight matrix की जगह दो छोटी matrices (B, A) train करता है — memory और समय दोनों कम। QLoRA base model को 4-bit में रखकर 4–8 GB GPU पर भी fine-tuning संभव बनाता है।
- **याद रखो:** *"Rank छोटा, असर बड़ा — LoRA।"*
- **Mini-project:** LoRA from scratch on a linear layer (NumPy/PyTorch), then PEFT LoRA on a small model in Colab.
- **Sources:** Hu et al., *LoRA* 2021 https://arxiv.org/abs/2106.09685 · Dettmers et al., *QLoRA* 2023 https://arxiv.org/abs/2305.14314 · Rafailov et al., *DPO* 2023 https://arxiv.org/abs/2305.18290 · PEFT https://huggingface.co/docs/peft

## Module 15 — Reinforcement Learning and AI Agents
- **Concepts:** agent/state/action/reward, MDPs, value functions, Bellman equation, Q-learning vs SARSA, DQN, policy gradients; LLM agents: tools, memory, planning, function calling, ReAct, structured outputs; guardrails (allow-listed tools, step/cost limits, human approval).
- **हिन्दी:** Q-learning भविष्य के सबसे अच्छे reward से सीखता है (off-policy), SARSA जो सच में किया उससे (on-policy)। Agent = LLM + tools + loop — हमेशा limits और approval के साथ।
- **याद रखो:** *"Agent को ताक़त दो, पर लगाम भी।"*
- **Mini-project:** Q-learning on a gridworld; a local research-assistant agent with 3 allow-listed tools and a step budget.
- **Sources:** Sutton & Barto, *Reinforcement Learning* http://incompleteideas.net/book/the-book-2nd.html · Mnih et al., *DQN* 2015 · Yao et al., *ReAct* 2022 https://arxiv.org/abs/2210.03629

## Module 16 — Generative and Multimodal AI
- **Concepts:** autoencoders, VAEs (reparameterisation, ELBO), GANs (generator vs discriminator), **diffusion**: forward noising, reverse denoising, DDPM objective, latent diffusion + text conditioning, CLIP, captioning, VQA.
- **हिन्दी:** Diffusion पहले image में धीरे-धीरे noise मिलाता है, फिर model उल्टा noise हटाना सीखता है। CLIP image और text को एक ही embedding space में लाता है।
- **याद रखो:** *"Noise जोड़ना आसान, हटाना सीखना ही generation है।"*
- **Mini-project:** tiny denoising autoencoder on digits; DDPM noise-schedule lab.
- **Sources:** Ho et al., *DDPM* 2020 https://arxiv.org/abs/2006.11239 · Kingma & Welling, *VAE* https://arxiv.org/abs/1312.6114 · Goodfellow et al., *GAN* https://arxiv.org/abs/1406.2661 · Radford et al., *CLIP* https://arxiv.org/abs/2103.00020 · Rombach et al., *Latent Diffusion* https://arxiv.org/abs/2112.10752

## Module 17 — MLOps, Deployment, Security and Responsible AI
- **Concepts:** project structure + config; experiment tracking (MLflow); pipelines data → deployment; serving with **FastAPI** + pydantic validation; Docker (non-root, pinned, health checks); monitoring (data/prediction drift, latency, LLM tracing); AI security (prompt injection, tool abuse, secrets); responsible AI (fairness metrics, privacy, model cards).
- **हिन्दी:** Model तभी काम का जब serve हो, monitor हो और rollback हो सके। Input validate करो, secrets कभी code में नहीं, drift पर alert।
- **याद रखो:** *"Deploy आसान, operate मुश्किल — monitoring पहले दिन से।"*
- **Mini-project:** Serve the Module 5 Titanic pipeline (joblib) with FastAPI `/predict` + `/health`, Dockerfile, MLflow-tracked training, and a drift check script.
- **Sources:** FastAPI https://fastapi.tiangolo.com/ · MLflow https://mlflow.org/docs/latest/ · OWASP Top 10 for LLM Apps https://genai.owasp.org/ · Mitchell et al., *Model Cards* https://arxiv.org/abs/1810.03993 · Google, *Rules of ML* https://developers.google.com/machine-learning/guides/rules-of-ml

## Module 18 — AI Researcher Track and Final Project
- **Concepts:** reading papers (3-pass), searching arXiv → code, baselines and ablations, multi-seed runs + statistical testing, reproducing at reduced scale, writing a report (claims ↔ evidence), peer review criteria, large-scale training (data/tensor/pipeline parallelism, ZeRO).
- **हिन्दी:** एक seed का result सबूत नहीं — कई seeds, confidence interval और baseline के साथ बताओ। हर claim के पीछे एक experiment।
- **याद रखो:** *"Claim छोटा, सबूत बड़ा।"*
- **Capstone:** reproduce one small result (e.g., LoRA rank vs accuracy) with 5 seeds and a 4-page report.
- **Sources:** Keshav, *How to Read a Paper* https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf · Henderson et al., *Deep RL that Matters* https://arxiv.org/abs/1709.06560 · Rajbhandari et al., *ZeRO* https://arxiv.org/abs/1910.02054

---

## Eight Capstone Systems (project specs)

| # | System | Core modules | Key deliverable |
|---|---|---|---|
| 1 | Fraud / risk ops | 6, 7, 8 | Leak-free features, calibrated risk model, alert queue with budget, FastAPI service |
| 2 | Demand forecasting + reorder policy | 9, 17 | Walk-forward backtest vs baseline, reorder rule, served forecast |
| 3 | RAG document assistant | 13, 17 | Grounded answers with citations, retrieval eval, injection defences |
| 4 | Hybrid recommender | 8, 17 | Cold-start handling, bias check, top-N API |
| 5 | Document extraction with confidence | 11, 12, 13 | Field extraction + confidence scores + human review queue |
| 6 | Vision detection + tracking monitor | 11 | Detection, IoU/NMS, event rules, frame monitor |
| 7 | Campus/helpdesk AI agent | 13, 15 | Tool-using agent with escalation rules and audit log |
| 8 | Fleet telemetry health model | 7, 9, 17 | Telemetry warehouse (SQL), health/delay model, drivers analysis |

Each capstone follows the same 5 lessons: **system tour → data/features without leakage → baseline vs model → serving → honest report.**

**याद रखो:** *हर project में baseline, leakage check, monitoring और ईमानदार report।*
