# ch-tts-llasa-rl-grpo

## Installation

You can follow the instructions in the [verl install](https://verl.readthedocs.io/en/latest/start/install.html).

## Data Preprocessing

### Push Dummy Dataset

[Push Dummy Dataset](./examples/data_preprocess/push_dummy_tts_ds.py) defines a dummy dataset and pushes it to the Hugging Face Hub.

### Make local hdfs directory

[Make local hdfs directory](./examples/data_preprocess/tts.py) makes a local hdfs directory and pushes the dataset to the hdfs directory.

```bash
python3 examples/data_preprocess/tts.py \
  --data_source Seungyoun/dummy_llasa_tts_text \
  --local_dir ~/data/llasa-tts-rl-grpo
```

## Training

This tutorial use 3xA6000 GPUs. 
2 for training and 1 for whisper nll calculation.

we make grpo reward objective function as follows:

Here's the reward calculation described clearly in English with mathematical notation:

---

### Reward Calculation

The reward is calculated based on two key metrics: the Character Error Rate (**CER**) and the Negative Log-Likelihood (**NLL**) obtained from Whisper. The formula is given by:

$$
\text{Reward} = \frac{\lambda_c + \lambda_n}{\frac{\lambda_c}{U_{CER}} + \frac{\lambda_n}{U_{NLL}}}
$$

where:

* **CER Utility**:

  $$
  U_{CER} = 1 - \tanh(\beta_c \cdot CER)
  $$

* **NLL Utility**:

  $$
  U_{NLL} = e^{-\frac{NLL}{\tau_n}}
  $$

---

### Explanation of Variables:

* **CER**: Character Error Rate (difference between the ground truth and Whisper's transcript).
* **NLL**: Negative Log-Likelihood from Whisper (a measure of speech synthesis quality).
* **$\beta_c$, $\tau_n$**: Parameters controlling sensitivity of CER and NLL respectively.
* **$\lambda_c$, $\lambda_n$**: Weights determining the relative importance of CER and NLL.

This results in a reward value ranging between **0 and 1**, with higher values indicating better quality.


## Launch Whisper server 

[Whisper server](./examples/whisper_server/whisper_server.py) is a server that calculates the NLL of the Whisper model.

```bash
CUDA_VISIBLE_DEVICES=2 \
python3 tts/whisper_server.py \
  --port 8001 \
  --model large-v3
```

then

```bash
WHISPER_SERVER=http://localhost:8001
```

## Launch Training

```bash
nohup bash ./examples/grpo_trainer/run_llasa_tts_grpo.sh > verl_grpo_1b.log 2>&1 &
```

### Results

<div align="center">
<img src="./misc/grpo_results.png" alt="results" width="70%">
</div>

We performed continual training of a Korean TTS model starting from the [LLASA-1B](https://huggingface.co/HKUSTAudio/Llasa-1B) checkpoint and evaluated its performance using our internal dataset.

The results clearly indicate an improvement when applying **GRPO**:

* **LLasa1B + 15K Korean**: CER = **0.0266**
* **LLasa1B + 15K Korean + GRPO**: CER = **0.0204**

The chart visually demonstrates that GRPO significantly reduces the Character Error Rate (CER), indicating enhanced synthesis quality.

# Acknowledge

This work is build upon [veRL](https://verl.readthedocs.io/en/latest/index.html) and [LLASA](https://arxiv.org/abs/2502.04128).