# TAFND

Official implementation of the ICONIP 2026 accepted paper:

**TAFND: Time-Aware Comment Propagation Network for Multi-Modal Fake News Detection**

TAFND is a multi-modal fake news detection framework that jointly models **news text**, **news images**, and **user comment propagation patterns**. In particular, TAFND explicitly incorporates the temporal information of comment propagation into a heterogeneous graph neural network to capture how discussion patterns evolve after a post is published.

The implementation supports experiments on **Fakeddit** and **Weibo**, including text-image baselines, comment-propagation baselines, and the proposed time-aware model.

---

## Repository Structure

```text
TAFND/
│
├── DataModule/
│   ├── data_preprocess.py
│   ├── load_data.py
│   ├── load_graph.py
│   ├── comment_graph_create_fakeddit.py
│   ├── comment_graph_create_fakeddit_time.py
│   ├── comment_graph_create_weibo_time.py
│   │
│   ├── data/
│   │   └── Pre-extracted .pt features and graph files
│   │
│   └── dataset/
│       ├── fakeddit/
│       └── weibo/
│
├── Models/
│   ├── concat_model.py
│   │   ├── Two-modal models
│   │   └── Non-temporal three-modal models
│   │
│   ├── temporal_model.py
│   │   └── TAFND models
│   │
│   └── trainer.py
│       └── Training and evaluation
│
├── TreeModule/
│   ├── hgt_propagation.py
│   │   └── Standard HGT propagation
│   │
│   └── hgt_propagation_time_aggr.py
│       └── Time-aware comment propagation + HGT
│
├── PretrainedModels/
│   └── Local pretrained Transformer models
│
├── Utils/
│   └── args.py
│
├── train_fakeddit_twomodal.py
├── train_fakeddit_threemodal.py
├── train_fakeddit_threemodal_time.py
│
├── train_weibo_twomodal.py
├── train_weibo_threemodal.py
├── train_weibo_threemodal_time.py
│
├── environment.yml
└── README.md
```

> **Note:** Python imports and data paths in the implementation use `DataModule`. On case-sensitive systems such as Linux, please make sure that the directory name is also exactly `DataModule`.

---

## Requirements

The experimental environment is provided in `environment.yml`.

The main dependencies include:

```text
Python 3.10
PyTorch 2.5.1
CUDA 12.1
PyTorch Geometric 2.6.1
torch-scatter 2.1.2
torch-sparse 0.6.18
Transformers 4.49.0
scikit-learn 1.6.1
```

The complete dependency list can be found in `environment.yml`.

Create the environment with:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate llm
```

---

## Pretrained Models

The current implementation loads pretrained models from the local directory:

```text
PretrainedModels/
```

The expected structure is approximately:

```text
PretrainedModels/
├── roberta-base/
├── chinese-roberta-wwm-ext/
└── vit-base-patch16-224-in21k/
```

For Fakeddit:

```text
Text  : roberta-base
Image : vit-base-patch16-224-in21k
```

For Weibo:

```text
Text  : chinese-roberta-wwm-ext
Image : vit-base-patch16-224-in21k
```

Make sure that the corresponding pretrained Hugging Face models are available under these directories before feature extraction.

---

## Data

The experiments are conducted on:

* **Fakeddit**
* **Weibo**

Due to the large size of the original datasets and the extracted `.pt` feature files, the complete processed datasets are not included directly in this repository.

A limited set of demo files is provided for reference.

The expected dataset structure is:

```text
DataModule/
├── dataset/
│   ├── fakeddit/
│   │   ├── data_json/
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   └── test.json
│   │   └── ...
│   │
│   └── weibo/
│       ├── data_json/
│       │   ├── train.json
│       │   ├── val.json
│       │   └── test.json
│       └── ...
│
└── data/
    ├── fakeddit_train.pt
    ├── fakeddit_test.pt
    ├── fakeddit_train_graph.pt
    ├── ...
    ├── weibo_train.pt
    └── ...
```

The preprocessing scripts automatically generate intermediate text/image embeddings and graph representations when the required processed files are unavailable.

---

## Running Experiments

### Fakeddit

#### Text + Image

```bash
python train_fakeddit_twomodal.py
```

#### Text + Image + Comment Graph

```bash
python train_fakeddit_threemodal.py
```

#### TAFND

```bash
python train_fakeddit_threemodal_time.py
```

---

### Weibo

#### Text + Image

```bash
python train_weibo_twomodal.py
```

#### Text + Image + Comment Graph

```bash
python train_weibo_threemodal.py
```

#### TAFND

```bash
python train_weibo_threemodal_time.py
```

---

## Evaluation

The implementation reports the following metrics:

```text
Accuracy
Precision
Recall
F1 Score
```

Metrics are separately reported for the fake-news and real-news classes.

For repeated experiments, results are reported in the form:

$$
\text{mean} \pm \text{standard deviation}.
$$

---

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{tafnd2026,
  title     = {TAFND: Time-Aware Comment Propagation Network for Multi-Modal Fake News Detection},
  booktitle = {International Conference on Neural Information Processing (ICONIP)},
  year      = {2026}
}
```

The complete BibTeX entry will be updated after the official publication information becomes available.

---

## Acknowledgements

This implementation is built with:

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pyg.org/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

We thank the authors and maintainers of these open-source projects.

---

## Contact

For questions regarding the implementation, please open an issue in this repository.
