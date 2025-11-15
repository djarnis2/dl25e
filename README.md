# FullyConnectedNN – BloodMNIST grid search

## Setup

```bash
# 1️⃣ Create virtual environment
python -m venv .venv

# 2️⃣ Activate environment (Windows)
.venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# Run baseline grid search to find best combinations
python -m scripts.grid1_baseline --epochs 12 --batch_size 128 --outdir outputs/grid1

This saves intermediate and final results (CSV files) under:
outputs/grid1/
    ├── grid1_partial.csv
    └── grid1_softmax_vs_hinge.csv

Figures are saved in:
# Generate plots from the grid search output
python -m scripts.plot_grid_results --csv outputs/grid1/grid1_softmax_vs_hinge.csv --outdir outputs/plots


outputs/plots/
