LLaSA Text-to-Speech GRPO Example
================================

Introduction
------------

This example demonstrates how to apply GRPO training to the
``LLaSA`` text-to-speech model. The reward is computed by transcribing
model outputs with ``whisper-turbo-v3`` and measuring the character error
rate (CER). The reward returned to the agent is ``1 - CER``, so the
training encourages lower transcription error.

Step 1: Prepare dataset
-----------------------

.. code:: bash

    python examples/data_preprocess/tts.py --local_dir ~/data/llasa-tts-rl

Step 2: Download model
----------------------

Download the LLaSA checkpoints from your preferred source (e.g.
HuggingFace) and set ``actor_rollout_ref.model.path`` in the run script
accordingly.

Step 3: Perform GRPO training
-----------------------------

.. code:: bash

    cd examples/grpo_trainer
    bash run_llasa_tts_grpo.sh

Make sure ``OPENAI_API_KEY`` is set so that ``whisper-turbo-v3`` can be
used for reward calculation.
