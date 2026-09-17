<h1 align="center">Hi, I'm Huy Nguyen 👋</h1>

<p align="center">
  <b>AI/ML Researcher & Engineer</b> · Hanoi University of Science and Technology
</p>

<p align="center">
  LLM Efficiency · Model Fingerprinting · Quantization · NLP Reasoning
</p>

---

### About me

I'm interested in making language models **smaller, more robust, and easier to understand**. My current work focuses on what happens to model behavior and embedded fingerprints when LLMs are quantized, modified, or evaluated under different inference settings.

I also work on **LLM reasoning for NLP**, especially logical fallacy detection and multi-agent evaluation.

### Selected LLM work

- **[LLM Fingerprint Quantization Attacks](https://github.com/HuyL13/quantization_attack)**  
  Utility-preserving quantization experiments designed to stress-test model fingerprints, with RTN baselines, learned/adversarial rounding variants, PPL evaluation, and upstream IF-SFT FSR verification.

- **[IF-SFT Fingerprint Forensics](https://github.com/HuyL13/phase1_if_analysis)**  
  Mechanistic analysis of how an IF-SFT fingerprint survives quantization across parameters, layers, margins, hidden representations, and quantization error structure.

- **[IF-SFT × VPTQ](https://github.com/HuyL13/if-sft-vptq-lab)**  
  Weight-only vector post-training quantization experiments on fingerprinted LLaMA-2-7B using VPTQ, with 3-bit/4-bit settings and official FSR evaluation.

- **[IF-SFT × TurboQuant](https://github.com/HuyL13/if-sft-turboquant-lab)**  
  Reproducible experiments testing whether TurboQuant KV-cache quantization changes IF-SFT fingerprint success under controlled inference settings.

- **[Multi-Agent Fallacy Detection](https://github.com/HuyL13/test_debate_agents)**  
  LLM-based fallacy detection and classification with factual, logical, and contextual agents, deliberation protocols, ablations, and reproducible evaluation on CoCoLoFa.

### Research interests

`LLM Quantization` · `Model Fingerprinting` · `Model Robustness` · `Efficient LLMs` · `NLP Reasoning` · `Multi-Agent Systems`

### Tools I use most

`Python` · `PyTorch` · `Hugging Face Transformers` · `CUDA` · `Linux` · `Docker`

---

<p align="center">
  <i>Building reproducible experiments around efficient, robust, and reasoning-capable language models.</i>
</p>
