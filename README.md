# BDA2025_FINAL

This repository is for BDA2025 Final Project, focusing on the usage of clustering method. The task is to analyze the relationships within this dataset and **classify the data into 4n – 1 clusters**, where n is the number of dimensions of the data.

The results will finally be evaluated based on the Fowlkes–Mallows Index (FMI), which measures the similarity between your clustering results and a hidden ground truth.

## Environment Setup

```
conda create --name bigdata python=3.10 
conda activate bigdata
pip install -r requirements.txt
```

## Single Run (Local)
- Use this mode when testing and developing a single configuration locally.
- Example usage:
```bash
python main.py public_data.csv public_submission.csv \
    --method gmm \
    --scaler power \
    --covariance tied \

python main.py private_data.csv private_submission.csv \
    --method gmm \
    --scaler power \
    --covariance tied \
```
- Other useful commands:
``` bash
python main.py --help   # Show command line options
python eval.py          # Evaluate FMI score for public_submission.csv
python plot.py          # Visualize dataset or clustering result
```

## Batch Run
- Use this mode when performing grid search across different hyperparameter combinations.
- Environment Used:
  - Platform: Kaggle Notebook
  - Accelerator: NVIDIA T4 GPU ×2
> Note: FMI results may vary across platforms. All results reported in this project were obtained using the above environment.
- How to Run:
  - Check and complete the TODO sections before execution:
    - `# TODO: Upload the dataset, then set the path here`
    - `# TODO: choose your hyperparameters here !!!`
- How to Evaluate:
  - After execution, `results.zip` and `config.txt` will be generated.
  - Download and extract them into `path_to_project/result/` folder.
  - Run the command as below:
```bash
python grade_all.py
```

## Folder Structure
```bash
path_to_project/
├── BDA_FINAL.pdf                     # Problem description
├── Report.pdf
├── README.md
├── requirements.txt                  # Dependencies

# Code files
├── main.py                           # Single-run script
├── bda2025-final.ipynb               # Batch-run notebook
├── eval.py                           # Evaluate FMI on public set
├── plot.py                           # Generate visualizations
├── grade_all.py                      # Evaluate all submissions
├── grader.cpython-310-*.so           # FMI scoring backend

# Datasets
├── public_data.csv
├── private_data.csv

# Submission outputs
├── public_submission.csv             # MY OWN single run results
├── private_submission.csv

# Visualizations
├── public_data_scatter_matrix.png
├── private_data_scatter_matrix.png
├── public_clustering_results.png
├── private_clustering_results.png

# Batch results
├── grid_result.txt                   # MY OWN batch run results
└── result/
    ├── config.txt                    # Hyperparameter config log
    ├── submission_1.csv
    ├── submission_2.csv
    └── ...                           # Up to submission_176.csv
```
