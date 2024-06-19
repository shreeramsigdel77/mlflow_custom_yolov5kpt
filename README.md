# MLflow Experiment Logging

This repository contains a script to log metrics, parameters, and artifacts from a machine learning experiment using MLflow. The script reads data from specific files and logs them into an MLflow experiment.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Requirements

- Python 3.7 or higher
- The following Python libraries:
  - `mlflow`
  - `pyyaml`

## Setup

* Create a virtual environment or use the existing virtual environment make sure (the pacages are not conflicting)

  ```
  python -m venv venv
  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
  ```
* **Clone the repository:**

  ```sh
  git clone <repository-url>
  cd <repository-directory>
  ```
* Install dependencies:

  ```
  pip install -r requirements.txt
  ```

## Usage

**Prepare your experiment files:**

* `hyp.yaml`: Contains hyperparameters.
* `results.txt`: Contains experiment results with metrics.
* `opt.yaml`: Contains training options.
* `weights/`: Directory containing model weights.

**Edit the `base_dir` variable in the script:**

* Set the `base_dir` variable in the `main` function to the path containing your experiment files.

**Run the script:**

```
python mlflow_script.py
```

**Start the MLflow UI:**

```
mlflow ui
```

Access the UI at `http://127.0.0.1:5000`.


## License

This project is licensed under the MIT License. See the [LICENSE]() file for details.
