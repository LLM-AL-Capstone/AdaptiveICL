
## Quick macOS (Apple Silicon) setup

## Quick macOS (Apple Silicon) setup

Notes for macOS (Apple Silicon):
- I have Updated the environment to be cross-platform and Apple Silicon–friendly. The original installation steps (by the author) has Linux-specific pins, so remove the partial env and recreate it using the updated `selective_annotation.yml`. (Follow steps below)
- The environment now uses Python 3.10 and installs PyTorch without CUDA (Metal/MPS will be used on macOS).
We follow the general setup of Votek (<https://github.com/xlang-ai/icl-selective-annotation>).

Instructions 

- Install Xcode Command Line Tools: `xcode-select --install`
- Recommended: Miniforge/Mambaforge (conda-forge native on Apple Silicon)
- Update conda and set channels (once):
  ```zsh
  conda update -n base -c defaults conda
  conda config --add channels conda-forge
  conda config --add channels pytorch
  conda config --add channels defaults
  conda config --set channel_priority strict
  ```

Create the environment
- Using conda (or replace “conda” with “mamba” for faster solves):
  ```zsh
  conda env create -f selective_annotation.yml
  conda activate selective_annotation
  ```

Quick compatibility fix (only if you see NumPy/PyArrow errors)
- If you hit “A module compiled with NumPy 1.x cannot run on NumPy 2.x” or a PyArrow error, run:
  ```zsh
  conda install -y -c conda-forge "numpy<2" "pyarrow>=14,<15"
  pip install --upgrade "datasets>=2.14"
  ```

Verify PyTorch + Metal (MPS)
```zsh
python - <<'PY'
import torch, numpy, pyarrow as pa, datasets as ds
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("pyarrow:", pa.__version__)
print("datasets:", ds.__version__)
print("MPS available (macOS):", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())
PY
```
- Expected on Apple Silicon: MPS available (macOS): True and CUDA available: False. This is correct and means PyTorch will use Apple’s Metal backend.

Minimal sanity run (lightweight, works on MPS/CPU)
```zsh
python main_adaptive_phases.py \
  --evaluate_calibration \
  --few_shot 2 \
  --task_name sst2 \
  --selective_annotation_method "ada_icl_default" \
  --model_cache_dir "models" \
  --data_cache_dir "datasets" \
  --output_dir outputs \
  --annotation_size 2 \
  --model_name "gpt2" \
  --seed 0 \
  --init "cluster" \
  --sample_k
```

Outputs
- Results are saved under `outputs/` (e.g., `outputs/adaptive_phases_few_shot-2-2/...`).
- Check `result_summary.txt` for metrics.

Tips
- Use `gpt2` for first run; larger models (e.g., gpt-neo 1.3B) may exceed laptop memory.
- If a previous broken env exists: `conda env remove -n selective_annotation`
- Faster solver: install mamba and use `mamba env create -f selective_annotation.yml`
- Optional: speed up HF downloads
  ```zsh
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_HUB_DISABLE_SYMLINKS_WARNING=1
  ```


# AdaICL: Which Examples to Annotate for In-Context Learning? Towards Effective and Efficient Selection

In this work, we investigate an active learning approach for ICL, where there is a limited budget for annotating examples. We propose a model-adaptive optimization-free algorithm, termed AdaICL, which identifies examples that the model is uncertain about, and performs semantic diversity-based example selection. Diversity-based sampling improves overall effectiveness, while uncertainty sampling improves budget efficiency and helps the LLM learn new information. Moreover, AdaICL poses its sampling strategy as a Maximum Coverage problem, that dynamically adapts based on the model’s feedback and can be approximately solved via greedy algorithms. 
![AdaICL algorithm.](assets/AdaICL_alg.pdf "AdaICL algorithm.")



## Installation
To establish the environment, run this code in the shell:
```
conda env create -f selective_annotation.yml
conda activate selective_annotation
```

## Usage

### Datasets

All datasets will be automatically downloaded from huggingface/datasets and stored here.

### End-to-end pipeline: selection, inference, evaluation
GPT-Neo as the in-context learning model, TREC and SST2 as the tasks, and AdaICL  as the selective annotation method, with additional budget of 20.
```
CUDA_VISIBLE_DEVICES=0 ./scripts/run_adaicl.sh
```

Example:
```
CUDA_VISIBLE_DEVICES=0 python main_adaptive_phases.py --evaluate_calibration --few_shot 5 --task_name ag_news --selective_annotation_method "ada_icl_default" --model_cache_dir "models" --data_cache_dir "datasets" --output_dir outputs --annotation_size 20 --model_name "gpt-neo" --seed 0 --init "cluster"  --sample_k 
```


## Directory Layout
Below you can find the scripts to reproduce the key results.

```bash
./active-in-context-learning
|---- MetaICL/                      # the model will be loaded similar to MetaICL for classification problems. That way we do not encouter invalid label generation.
|---- logs/                         # Folder for storing logfiles.
|---- outputs/                      # Folder for storing output results.
|---- scripts/                      # Run these scripts to reproduce results.
|
|---- algorithms.py                 # k-means, fast-votek, model_uncertainty_estimation, votek utilies
|---- annotation_methods.py         # Supported active learning algos.
|---- get_task.py                   # Dataset-specific utilies.
|---- main_adaptive_phases.py       # Execution of AL algos in an adaptive manner (inductive).
|---- main_generative.py            # Generation tasks.
|---- prompt_retrieval.py           # Retrieve prompts from annotated pool.
|---- utils.py                      # BERT embeddings, plots, calibration error etc.
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.