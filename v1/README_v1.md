# NeSyWare v1 - MalImg Release

A command-line implementation of the NeSyWare framework for static malware family classification, combining a deep CNN with a Logic Tensor Network (LTN) knowledge base.  
Trained on the **MalImg** dataset; classifies 24 malware families from binary visualisations of PE files or raw PNG images.

---

## Architecture

```
Input
  │  PNG image (MalImg-style)   or   PE file → Nataraj-2011 binary image
  ▼
MalImgCNN  (grayscale, 4-block custom CNN)
  │  feature_dim = 8 192  (512 × 4 × 4 via AdaptiveAvgPool)
  ├─► Neural classifier head  →  logits₂₄
  └─► SymbolicPredicateGrounding  →  29 behavioural predicates ∈ [0,1]
                                         │
                                         ▼
                              Knowledge Base (16 expert IF-THEN rules)
                              fuzzy AND/OR → KB scores₂₄ ∈ [0,1]
  │
  ▼
Combined score = 0.70 × neural_logits + 0.30 × log(KB_scores + ε)
  │
  ▼
softmax  →  probability distribution over 24 families
```

### Supported families

24 families across 8 categories: Adware, Backdoor, Dialer, Generic Malware, Obfuscator, Password Stealer, Rogue/FakeAV, Trojan, and Worm.  
Full family list available in `families.md`.

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- Pillow ≥ 10.0
- NumPy ≥ 1.24

```bash
pip install -r requirements.txt
```

The weights file (`weights/malimg_nesyware.pth`, ~900 MB) must be present before running.

| File | Description | Download |
|------|-------------|----------|
| `weights/malimg_nesyware.pth` | Model checkpoint (~900 MB) | [HuggingFace](https://huggingface.co/lostCosmonaut/NeSyWare/tree/main/weights_v1) |

---

## Usage

One or more files can be passed as arguments:

```bash
python analyze.py sample.exe
python analyze.py sample.png sample2.exe sample3.dll
```

Accepted input types:

| Type | Extensions |
|------|-----------|
| MalImg-style image | `.png`, `.bmp`, `.jpg`, `.jpeg`, `.tif`, `.tiff` |
| PE executable | any other extension (`.exe`, `.dll`, `.sys`, …) |

Example output:

```
============================================================
File: sample.exe
Result  : [HIGH_CONFIDENCE] Lolyda.AA1  (87.4%)  [Password Stealer]
Profile :
   1. Lolyda.AA1             ████████████████████████████████  87.4%
   2. Lolyda.AA2             ███                                7.1%
Predicates (active):
  Credential Theft               0.91
  Registry Modification          0.83
  Network Activity               0.61
```

Confidence levels:

| Level | Threshold | Meaning |
|-------|-----------|---------|
| HIGH CONFIDENCE | top-1 ≥ 70% | Single dominant family identified |
| MODERATE | top-1 50–70% | Likely match, some ambiguity |
| AMBIGUOUS | top-1 20–50% | Multiple candidates |
| INCONCLUSIVE | top-1 < 20% | Possibly unseen or heavily obfuscated family |

---

## Training

### Dataset

The [MalImg dataset](https://dl.acm.org/doi/10.1145/2016904.2016908) (Nataraj et al., 2011) contains grayscale PNG visualisations of PE files where each pixel encodes one byte of the binary.  
This model was trained on a 24-class subset (one class excluded due to cross-contamination in the available split).

| Split | Samples |
|-------|---------|
| Train (80%) | ~7 450 |
| Validation (20%) | ~1 860 |

Difficult classes received augmentation with heavy geometric and photometric transforms.

### Model

- **Backbone**: 4-block CNN (64→128→256→512 channels), BatchNorm + Dropout, AdaptiveAvgPool → 8 192-dim feature vector
- **Classifier head**: FC 8192→1024→512→24
- **Predicate grounding**: 29 independent MLPs, one per behavioural predicate
- **Knowledge base**: 16 expert IF-THEN rules using fuzzy product T-norm → per-class activation scores ∈ [0, 1]

### Results

| Metric | Value |
|--------|-------|
| Best validation accuracy | **98.73%** |
| Final validation accuracy | 98.24% |
| Best accuracy over Axioms' ablation (5 out 7) | 99.16% |

---

## Files

| File | Description |
|------|-------------|
| `analyze.py` | Main entry point - CLI classifier |
| `inference.py` | Self-contained inference engine |
| `pe_to_image.py` | PE → Nataraj-2011 grayscale image converter |
| `requirements.txt` | Python dependencies |
| `weights/malimg_nesyware.pth` | Model checkpoint (~900 MB) |
| `families.md` | Full list of supported families and categories |

---

## Limitations

- Trained exclusively on MalImg-24 - samples from outside this distribution will likely produce inconclusive or ambiguous results.
- Static analysis only: packed, encrypted, or obfuscated binaries that differ visually from their training representatives may be misclassified.
- The 29 symbolic predicates are grounded from visual features, not dynamic execution traces; they reflect visual correlates of behaviour, not verified runtime actions.
- No binary gatekeeper stage: the model classifies directly into one of the 24 malware families without a preliminary benign vs. malware discrimination step. Benign PE files fed to this version will receive an arbitrary family label rather than being rejected.

---

## Future Improvements

The following improvements are currently under development and will be addressed in upcoming versions of NeSyWare:

- **Binary gatekeeper stage** - a dedicated first stage that separates benign from malicious samples before any family classification takes place, making the pipeline robust to real-world inputs that include clean PE files.
- **Multi-dataset blending** - extending training beyond MalImg-24 by incorporating multiple well-known malware datasets, increasing both the number of supported families and the generalisability of the learned representations across different sample sources and collection periods.

> ⚠️ **Disclaimer** - NeSyWare is an AI-based research framework and may produce
> incorrect classifications. It is intended to assist - NOT replace - professional
> malware analysis workflows. Results should always be cross-validated with
> VirusTotal, sandbox environments, and dedicated static/dynamic analysis tools.
