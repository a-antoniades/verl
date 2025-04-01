<<<<<<< HEAD
### FORK and DEBUG with Agent Data

To run the project with debugging enabled and agent data, execute the following command in your terminal:
```bash
bash examples/grpo_trainer/run_qwen2-7b_seq_balance_debug.sh
```
After running the project, you can view the data using the Jupyter notebook located at `notebooks/view_data.ipynb`.


<h1 style="text-align: center;">veRL: Volcano Engine Reinforcement Learning for LLM</h1>
<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLM</h1>

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
![GitHub forks](https://img.shields.io/github/forks/volcengine/verl)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
<a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
![GitHub contributors](https://img.shields.io/github/contributors/volcengine/verl)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>


verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid-controller programming model enables flexible representation and efficient execution of complex Post-Training dataflows. Build RL dataflows such as GRPO, PPO in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, SGLang, etc

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular HuggingFace models


verl is fast with:

- **State-of-the-art throughput**: SOTA LLM training and inference engine integrations and SOTA RL throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

</p>

## News
- [2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is [publicly available](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main/recipe/dapo) now.
- [2025/03] We will present verl(HybridFlow) at EuroSys 2025. See you in Rotterdam!
- [2025/03] We introduced the programming model of verl at the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg) and [verl intro and updates](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl-lmsys-meetup.pdf) at the [LMSys Meetup](https://lu.ma/ntjrr7ig) in Sunnyvale mid March.
- [2025/02] verl v0.2.0.post2 is released! See [release note](https://github.com/volcengine/verl/releases/) for details.
- [2025/01] [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro) is released with SOTA-level performance on LLM & VLM. The RL scaling preview model is trained using verl, reaching OpenAI O1-level performance on math benchmarks (70.0 pass@1 on AIME).
<details><summary> more... </summary>
<ul>
  <li>[2025/02] We presented verl in the <a href="https://lu.ma/ji7atxux">Bytedance/NVIDIA/Anyscale Ray Meetup</a>. See you in San Jose!</li>
  <li>[2024/12] verl is presented at Ray Forward 2024. Slides available <a href="https://github.com/eric-haibin-lin/verl-community/blob/main/slides/Ray_Forward_2024_%E5%B7%AB%E9%94%A1%E6%96%8C.pdf">here</a></li>
  <li>[2024/10] verl is presented at Ray Summit. <a href="https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37">Youtube video</a> available.</li>
  <li>[2024/12] The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024. <a href="https://github.com/eric-haibin-lin/verl-data/tree/neurips">Slides</a> and <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">video</a> available.</li>
  <li>[2024/08] HybridFlow (verl) is accepted to EuroSys 2025.</li>
</ul>	
</details>

## Key Features

- **FSDP** and **Megatron-LM** for training.
- **vLLM**, **SGLang**(experimental) and **HF Transformers** for rollout generation.
- Compatible with Hugging Face Transformers and Modelscope Hub: Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc
- Supervised fine-tuning.
- Reinforcement learning with [PPO](examples/ppo_trainer/), [GRPO](examples/grpo_trainer/), [ReMax](examples/remax_trainer/), [Reinforce++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm), [RLOO](examples/rloo_trainer/), [PRIME](recipe/prime/), etc.
  - Support model-based reward and function-based reward (verifiable reward)
  - Support vision-language models (VLMs) and [multi-modal RL](examples/grpo_trainer/run_qwen2_5_vl-7b.sh)
- Flash attention 2, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [sequence parallelism](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh).
- Scales up to 70B models and hundreds of GPUs.
- Experiment tracking with wandb, swanlab, mlflow and tensorboard.

## Upcoming Features
- DeepSeek 671b optimizations with Megatron v0.11
- Multi-turn rollout optimizations

## Getting Started

<a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a>

**Quickstart:**
- [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

**Running a PPO example step-by-step:**
- Data and Reward Preparation
  - [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- Understanding the PPO Example
  - [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [Run GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**Reproducible algorithm baselines:**
- [PPO, GRPO, ReMax](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**For code explanation and advance usage (extension):**
- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
- Advance Usage and Extension
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)

**Blogs from the community**
- [使用verl进行GRPO分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
- [HybridFlow veRL 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
- [最高提升20倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)


## Performance Tuning Guide
The performance is essential for on-policy RL algorithm. We write a detailed performance tuning guide to allow people tune the performance. See [here](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for more details.

## Use vLLM v0.8
veRL now supports vLLM>=0.8.0 when using FSDP as the training backend. Please refer to [this document](docs/README_vllm0.8.md) for installation guide and more information.

## Citation and acknowledgement

If you find the project helpful, please cite:
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```tex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and supported by Anyscale, Bytedance, LMSys.org, Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, and many more.

## Awesome work using verl
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [DAPO](https://dapo-sia.github.io/): the fully open source SOTA RL algorithm that beats DeepSeek-R1-zero-32B ![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [deepscaler](https://github.com/agentica-project/deepscaler): iterative context scaling with GRPO ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/deepscaler)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Logic-RL](https://github.com/Unakar/Logic-RL): a reproduction of DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset. ![GitHub Repo stars](https://img.shields.io/github/stars/Unakar/Logic-RL)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): Hacking **Real Search Engines** and **retrievers** with LLMs via RL for **information retrieval** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [cognitive-behaviors](https://github.com/kanishkg/cognitive-behaviors): Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs ![GitHub Repo stars](https://img.shields.io/github/stars/kanishkg/cognitive-behaviors)
- [MetaSpatial](https://github.com/PzySeere/MetaSpatial): Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse ![GitHub Repo stars](https://img.shields.io/github/stars/PzySeere/MetaSpatial)
- [DeepEnlighten](https://github.com/DolbyUUU/DeepEnlighten): Reproduce R1 with **social reasoning** tasks and analyze key findings ![GitHub Repo stars](https://img.shields.io/github/stars/DolbyUUU/DeepEnlighten)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [self-rewarding-reasoning-LLM](https://arxiv.org/pdf/2502.19613): self-rewarding and correction with **generative reward models** ![GitHub Repo stars](https://img.shields.io/github/stars/RLHFlow/Self-rewarding-reasoning-LLM)
- [critic-rl](https://github.com/HKUNLP/critic-rl): LLM critics for code generation ![GitHub Repo stars](https://img.shields.io/github/stars/HKUNLP/critic-rl)
- [DQO](https://arxiv.org/abs/2410.09302): Enhancing multi-Step reasoning abilities of language models through direct Q-function optimization
- [FIRE](https://arxiv.org/abs/2410.21236): Flaming-hot initiation with regular execution sampling for large language models

## Contribution Guide
Contributions from the community are welcome! Please checkout our [roadmap](https://github.com/volcengine/verl/issues/22) and [release plan](https://github.com/volcengine/verl/issues/354).

### Code formatting
We use yapf (Google style) to enforce strict code formatting when reviewing PRs. To reformat you code locally, make sure you installed **latest** `yapf`
```bash
pip3 install yapf --upgrade
```
Then, make sure you are at top level of verl repo and run
```bash
bash scripts/format.sh
```
We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in MLSys/LLM reasoning/multimodal alignment.
=======
<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![](https://img.shields.io/badge/Gurubase-(experimental)-006BFF)](https://gurubase.io/g/sglang)

</div>

--------------------------------------------------------------------------------

| [**Blog**](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
| [**Documentation**](https://docs.sglang.ai/)
| [**Join Slack**](https://slack.sglang.ai/)
| [**Join Bi-Weekly Development Meeting**](https://meeting.sglang.ai/)
| [**Roadmap**](https://github.com/sgl-project/sglang/issues/4042)
| [**Slides**](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides) |

## News
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] 🔥 SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] 🔥 v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, continuous batching, token attention (paged attention), speculative decoding, tensor parallelism, chunked prefill, structured outputs, and quantization (FP8/INT4/AWQ/GPTQ).
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, QWen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

## Getting Started
- [Install SGLang](https://docs.sglang.ai/start/install.html)
- [Quick Start](https://docs.sglang.ai/backend/send_request.html)
- [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
- [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap
[Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship
The project has been deployed to large-scale production, generating trillions of tokens every day.
It is supported by the following institutions: AMD, Atlas Cloud, Baseten, Cursor, DataCrunch, Etched, Hyperbolic, Iflytek, Jam & Tea Studios, LinkedIn, LMSYS, Meituan, Nebius, Novita AI, NVIDIA, RunPod, Stanford, UC Berkeley, UCLA, xAI, and 01.AI.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us

For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgment and Citation
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql). Please cite the paper, [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), if you find the project useful.
>>>>>>> sglang-fork/feature/http_server_engine
