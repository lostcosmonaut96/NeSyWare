# NeSyWare
**A Neuro-Symbolic Framework for Hierarchical PE Malware Classification**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)


---
> 📝 **Paper under review at ESORICS '26.** Citation will be updated upon acceptance.
---

## Overview

NeSyWare is a research framework that combines **deep learning** with **symbolic reasoning** to classify Windows PE malware in an interpretable, axiom-constrained manner.

Most malware classifiers are black boxes: they produce a label, but give no insight into *why* a sample belongs to a given family. NeSyWare is designed to bridge that gap - producing not just a prediction, but a structured, human-readable symbolic justification grounded in behavioral semantics.

---

## How It Works

### Binary Representation

PE binaries are transformed into a structured representation using well-established computer vision techniques. This step requires no manual feature engineering: the transformation preserves structural regularities inherent in the binary layout, such as section boundaries, packing artifacts, and data distribution across the file.

### Neural Feature Extraction

A deep neural network processes the binary representation and produces a compact latent feature vector. The architecture is chosen for its effectiveness at capturing hierarchical visual patterns, and is trained to distinguish malware from benign software before any symbolic reasoning takes place.

### Neuro-Symbolic Reasoning

The core contribution of NeSyWare is a **symbolic reasoning layer** built on **Logic Tensor Networks (LTNs)** - a framework that expresses logical axioms as differentiable constraints, trainable end-to-end alongside the neural backbone.

Classification proceeds in a **hierarchical cascade**:

- **Category level** - the model assigns a high-level behavioral category (e.g., Trojan, Ransomware, Worm) by enforcing category-level symbolic axioms
- **Family level** - within each category, family-specific predicates further refine the classification, grounded in behavioral traits derived from documented API call patterns and known malware characteristics

Each prediction is accompanied by a **predicate activation profile**: a set of symbolic, human-readable statements that explain which behavioral traits drove the classification decision.

#### Why Neuro-Symbolic?

| Property | Pure Neural | NeSyWare |
|----------|-------------|----------|
| Accuracy | ✅ High | ✅ High |
| Interpretability | ❌ Black box | ✅ Symbolic explanation |
| Behavioral grounding | ❌ None | ✅ Behavioral predicates |
| Knowledge extensibility | ❌ Retrain | ✅ More KB axioms → richer predicate profile; family separation requires further axiom refinement |
| Analyst-readable output | ❌ Logit vector | ✅ Predicate activation profile |

---

## Pipeline Architecture

The framework is organized as a cascaded multi-stage pipeline. Each version of NeSyWare documents its own concrete architecture - the schema below describes the general design:

```
┌──────────────────────────────────────────────┐
│              INPUT: PE Binary                │
└──────────────────┬───────────────────────────┘
                   │
         [ Binary Representation ]
           (version-specific)
                   │
          ┌────────▼─────────┐
          │  Neural Encoder  │  ← feature extraction
          └────────┬─────────┘
                   │
          ┌────────▼─────────┐
          │  LTN - Category  │  ← symbolic axioms (category level)
          └────────┬─────────┘
                   │
          ┌────────▼─────────┐
          │  LTN - Family    │  ← symbolic predicates (family level)
          └────────┬─────────┘
                   │
          ┌────────▼──────────────────┐
          │  Symbolic Explanation     │
          │  Predicate activations    │
          │  + behavioral profile     │
          └───────────────────────────┘
```

---

## Versions

| Version | Description |
|---------|-------------|
| [v1](./v1/README_v1.md) | Single-stage family classifier on MalImg-24. Custom CNN backbone, 29 behavioral predicates, no binary gatekeeper. |
| [v2](./v2/README_v2.md) | Three-stage hierarchical pipeline. ResNet-50 backbone, binary gatekeeper (Stage 1), 117 families across 10 categories, 54 symbolic predicates. Multi-dataset training with SOREL auxiliary supervision. |

---

## Data Sources

### Malware Samples

Malware samples are sourced from publicly available, well-known datasets commonly used in the academic malware analysis community. Each version of NeSyWare specifies which datasets were used, along with the labeling pipeline applied to normalize family names.

### Benign Samples

Benign PE files are collected from a **clean QEMU Windows 11 virtual machine**. The base installation is kept unmodified, and a curated set of third-party applications is subsequently installed and verified. Each binary is individually verified to have **zero detections on VirusTotal** before being included in the dataset.

---

## Contributing

NeSyWare is designed to be **dataset-agnostic**. Contributions are welcome in two forms: new malware datasets, and behavioral knowledge about existing families.

### Contributing Malware Datasets

Please ensure your dataset follows this structure:

```
dataset/
├── metadata.csv
└── malware/
    ├── <family_name_1>/
    │   └── *.exe / *.dll
    ├── <family_name_2>/
    │   └── *.exe / *.dll
    └── ...
```

#### `metadata.csv` Schema

| Column | Type | Description |
|--------|------|-------------|
| `sha256` | string | SHA-256 hash of the binary |
| `family` | string | AVClass-normalized family label |
| `category` | string | High-level category (e.g., `Trojan`, `Ransomware`) |
| `source` | string | Dataset origin (e.g., `MalwareBazaar`, `VirusTotal`) |

#### Minimum Requirements
- Malware samples should be at least described by category
- **≥ 80 samples per family** for reliable symbolic grounding
- Family labels should be **AVClass-normalized**
- Binaries must be **Windows PE** (`MZ` magic bytes)

> ⚠️ **Do not include malware binaries directly in the PR.** Share only `metadata.csv` and document where the binaries can be obtained (e.g., MalwareBazaar, VirusTotal, VirusShare).

### Contributing Behavioral Knowledge

If you have expertise on specific malware families, you can contribute **behavioral descriptions** that help refine the symbolic predicate layer. For each family, provide a structured description covering:

- Known API call patterns (e.g., file system, network, registry, process)
- Observed behavioral traits (e.g., persistence mechanisms, evasion techniques, payload delivery)
- Any relevant references (sandbox reports, threat intelligence, academic papers)

This information is used to refine or add LTN axioms grounding the predicate layer, improving symbolic separability between families without requiring additional training data.

### Contributing Benign Application Lists

Rather than submitting benign binaries directly, you can contribute a **curated list of legitimate Windows applications** to be used as benign sources. For each application, provide:

- Application name and version
- Official download source (vendor URL)
- SHA-256 hashes of the main executables

All listed binaries will be independently verified against VirusTotal before inclusion.

>Accepted datasets will be considered for inclusion in future training runs and acknowledged in the corresponding release notes.

---

## Citation

If you use NeSyWare in your research, please cite:

```bibtex
@inproceedings{nesyware2025,
  title     = {NeSyWare: A Neuro-Symbolic Framework   for Hierarchical PE Malware Classification via Differentiable Behavioral Predicates over Binary Images},
  author    = {Basciano, P.M. and Farina, G. and Monteleone, S.},
  booktitle = {Under review},
  year      = {2025}
}
```

Furthermore, if you use NeSyWare in your research, please also cite the datasets on which it was trained. Full BibTeX entries are available in [`CITATIONS.bib`](./CITATIONS.bib).

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

> ⚠️ **Disclaimer** - NeSyWare is an AI-based research framework and may produce
> incorrect classifications. It is intended to assist - NOT replace - professional
> malware analysis workflows. Results should always be cross-validated with
> VirusTotal, sandbox environments, and dedicated static/dynamic analysis tools.
