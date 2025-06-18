# phm_ml
Apply heuristic/statistical/machine learning/deep learning methods on PHM (Prognostic Health Management) in industrial manufacturing.

## Environment Setup using uv

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management and virtual environment handling. All following instructions are for macOS users, for Windows users, please refer to the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) or ask ChatGPT ^_^.

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/liuzi/phm_ml.git
   cd phm_ml
   ```

3. **Create virtual environment and install dependencies**:
   ```bash
   uv sync
   ```

   This will automatically:
   - Create a virtual environment
   - Install all dependencies from `pyproject.toml`
   - Install the project in editable mode

4. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate 
   ```

## Data Setup

1. **Place your data files** in the `data/interim/` directory: #TODO: add scripts and instructions for converting raw data to interim data
   - Training data: `Lab1-2017-Q1-ST4000DM000.parquet`
   - Test data: `Lab1-2016-Q4Half-ST4000DM000.parquet`

2. **Configure data paths** in `data.yaml`:
   ```yaml
   path:
     base_directory: "data" 
     train_data: "interim/Lab1-2017-Q1-ST4000DM000.parquet"
     test_data: "interim/Lab1-2016-Q4Half-ST4000DM000.parquet"
   ```

## Usage

### Running the Main Script

Execute the Backblaze hard drive data processing script:

```bash
python scripts/run_backblaze.py
```

This script will:
- Load parquet data from the specified path
- Apply data cleaning using the `phm_ml.data_process.data_backblaze_harddrive` module
- Display the first 50 rows of processed data


