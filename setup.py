# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate==1.0.1",
    "codetiming==1.4.0",
    "datasets==2.20.0",
    "dill==0.3.8",
    "hydra-core==1.3.2",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "peft==0.14.0",
    "pyarrow==19.0.0",
    "pybind11==2.13.6",
    "pylatexenc==2.10",
    "ray[default]==2.41.0",
    "torchdata==0.11.0",
    "tensordict==0.6.2",
    "transformers==4.51.0",
    "wandb==0.18.7",
    "packaging==24.1",
]

TEST_REQUIRES = ["pytest==8.3.3", "pre-commit==4.0.1", "py-spy==0.4.0"]
PRIME_REQUIRES = ["pyext==0.7"]
GEO_REQUIRES = ["mathruler==0.1.0"]
GPU_REQUIRES = ["liger-kernel==0.5.4", "flash-attn==2.7.4.post1"]
MATH_REQUIRES = ["math-verify==0.5.2"]  # Add math-verify as an optional dependency
VLLM_REQUIRES = ["tensordict==0.6.2", "vllm==0.8.5"]
SGLANG_REQUIRES = ["tensordict==0.6.2", "sglang[srt,openai]==0.4.6.post4", "torch-memory-saver==0.0.5", "torch==2.6.0"]

extras_require = {
    "test": TEST_REQUIRES,
    "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/volcengine/verl",
    license="Apache 2.0",
    author="Bytedance - Seed - MLSys",
    author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk",
    description="verl: Volcano Engine Reinforcement Learning for LLM",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
