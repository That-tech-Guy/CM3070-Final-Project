# Two-Stage Learning Resource Recommender for CAPE Computer Science

This repository contains the implementation developed for the CM3070 Final Project. It is a controlled proof of concept that recommends learning material for ten selected topics from CAPE Computer Science Unit 1.


The recommender makes two related decisions:

1. **Stage 1 - topic ranking:** ranks all ten candidate topics to identify what the learner should study next
2. **Stage 2 - resource ranking:** ranks the four curated resources associated with the selected topic

Synthetic dataset creation is a preparation step and is not an additional recommender stage

## Important interpretation boundary

The project was evaluated using synthetic learner interactions and simulator-defined outcomes. The results demonstrate the behaviour, technical correctness and comparative performance of the implemented pipeline under the disclosed assumptions. They do not establish improved learning, retention or classroom effectiveness for real CAPE students.

The generated educator-review CSV is a blank protocol for future work. It is not a completed educator evidence at this point

## Repository contents

| File or folder | Purpose |
| --- | --- |
| `Complete_Recommender_Pipeline.ipynb` | Recommended entry point that runs the complete workflow in one notebook |
| `Stage1_generate_synthetic_dataset.ipynb` | Generates and validates the synthetic dataset independently |
| `Stage2_model_training_and_evaluation.ipynb` | Builds candidates, trains models and creates the evaluation outputs from the Stage 1 dataset |
| `generate_dataset.py` | Dataset generation, graph processing, text processing, validation and output functions |
| `build_two_stage_candidates.py` | Reconstructs the information available before each historical recommendation event |
| `model_components.py` | Shared preprocessing, Ridge, Random Forest and matrix-factorisation components |
| `two_stage_recommender.py` | Tunes models, scores candidates, produces recommendations and saves evaluation results |
| `validate_two_stage_outputs.py` | Checks candidate counts, leakage controls, output files and saved models |
| `config.json` | Dataset scale, fixed seed and disclosed simulation assumptions |
| `seed_data/` | Researcher-prepared topic, prerequisite, resource and methodological-source tables |
| `requirements.txt` | Python package versions used by the project |
| `two_stage_results/` | Lightweight evaluation tables, figures and summaries retained as report evidence |

The notebooks use  the reusable Python modules above


## Generated folders

Running the complete pipeline creates or updates the following folders:

| Folder | Generated contents |
| --- | --- |
| `generated_output/` | Public synthetic data, separated simulator truth and dataset-validation evidence |
| `two_stage_data/` | Public topic and resource candidate tables |
| `two_stage_internal_truth/` | Evaluation-only oracle tables and the educator-review method key |
| `two_stage_validation/` | Candidate-construction validation results |
| `two_stage_results/` | Regression, ranking, end-to-end, cold-start, ablation, sensitivity and statistical results |
| `two_stage_saved_models/` | Fitted Ridge, Random Forest and matrix-factorisation models |
| `educator_validation/` | Blank blinded synthetic-case review form for possible future educator assessment |

The large generated datasets and model files are excluded from Git because they can be reconstructed from the committed source files


## Requirements

- Windows, macOS or Linux
- Python 3
- Enough memory and free disk space to construct the candidate tables
- The packages pinned in `requirements.txt`

The project was made on a Windows x64 maching and using Python 3.14.4. Package versions, rather than the Python installation itself, are pinned in `requirements.txt`.


## Clone the repository

```bash
git clone https://github.com/That-tech-Guy/CM3070-Final-Project.git
cd CM3070-Final-Project
```


## Set up the environment

### Windows PowerShell or Command Prompt

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```


## Recommended execution route

1. Open `Complete_Recommender_Pipeline.ipynb` in JupyterLab
2. Select **Restart Kernel and Run All Cells**
3. Allow the long-running generation and modelling cells to finish
4. Confirm that the final technical-validation checks pass

The complete notebook presents seed inspection, synthetic dataset generation, time-safe candidate construction, model training, recommendation generation and evaluation in one sequence.



