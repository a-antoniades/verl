<<<<<<< HEAD
Welcome to verl's documentation!
================================================

.. _hf_arxiv: https://arxiv.org/pdf/2409.19256

verl is a flexible, efficient and production-ready RL training framework designed for large language models (LLMs) post-training. It is an open source implementation of the `HybridFlow <hf_arxiv>`_ paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping and parallelism**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular HuggingFace models


verl is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, verl achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

--------------------------------------------

.. _Contents:

.. toctree::
   :maxdepth: 5
   :caption: Quickstart

   start/install
   start/quickstart
   start/multinode

.. toctree::
   :maxdepth: 4
   :caption: Programming guide

   hybrid_flow

.. toctree::
   :maxdepth: 5
   :caption: Data Preparation

   preparation/prepare_data
   preparation/reward_function

.. toctree::
   :maxdepth: 5
   :caption: Configurations

   examples/config

.. toctree::
   :maxdepth: 2
   :caption: PPO Example

   examples/ppo_code_architecture
   examples/gsm8k_example

.. toctree:: 
   :maxdepth: 1
   :caption: PPO Trainer and Workers

   workers/ray_trainer
   workers/fsdp_workers
   workers/megatron_workers

.. toctree::
   :maxdepth: 1
   :caption: Performance Tuning Guide
   
   perf/perf_tuning
   README_vllm0.7.md
   README_vllm0.8.md

.. toctree::
   :maxdepth: 1
   :caption: Experimental Results

   experiment/ppo

.. toctree::
   :maxdepth: 1
   :caption: Advance Usage and Extension

   advance/placement
   advance/dpo_extension
   advance/fsdp_extension
   advance/megatron_extension

.. toctree::
   :maxdepth: 1
   :caption: API References

   data.rst


.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq/faq

Contribution
-------------

verl is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/volcengine/verl>`_, `Slack <https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA>`_ and `Wechat <https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG>`_ for discussions.

Code formatting
^^^^^^^^^^^^^^^^^^^^^^^^
We use yapf (Google style) to enforce strict code formatting when reviewing MRs. Run yapf at the top level of verl repo:

.. code-block:: bash

   pip3 install yapf
   yapf -ir -vv --style ./.style.yapf verl examples tests
=======
SGLang Documentation
====================

SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, continuous batching, token attention (paged attention), speculative decoding, tensor parallelism, chunked prefill, structured outputs, and quantization (FP8/INT4/AWQ/GPTQ).
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, QWen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   start/install.md

.. toctree::
   :maxdepth: 1
   :caption: Backend Tutorial

   references/deepseek
   backend/send_request.ipynb
   backend/openai_api_completions.ipynb
   backend/openai_api_vision.ipynb
   backend/openai_api_embeddings.ipynb
   backend/native_api.ipynb
   backend/offline_engine_api.ipynb
   backend/server_arguments.md
   backend/sampling_params.md
   backend/hyperparameter_tuning.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   backend/speculative_decoding.ipynb
   backend/structured_outputs.ipynb
   backend/function_calling.ipynb
   backend/separate_reasoning.ipynb
   backend/custom_chat_template.md
   backend/quantization.md

.. toctree::
   :maxdepth: 1
   :caption: Frontend Tutorial

   frontend/frontend.ipynb
   frontend/choices_methods.md

.. toctree::
   :maxdepth: 1
   :caption: SGLang Router

   router/router.md

.. toctree::
      :maxdepth: 1
      :caption: References

      references/general
      references/hardware
      references/advanced_deploy
      references/performance_tuning
>>>>>>> sglang-fork/feature/http_server_engine
