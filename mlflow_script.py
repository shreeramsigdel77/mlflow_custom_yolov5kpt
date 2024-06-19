import mlflow
import yaml
import os

def is_header(line:str)->bool:
    """
    Determine if a given line is a header based on the ratio of non-numeric to numeric values.
    :param line: str, a single line from the file
    :return: bool, True if the line is likely a header, False otherwise
    """
    # Split the line into parts
    parts = line.split()
    num_numeric = sum(part.replace(".", "").replace("-", "").isdigit() for part in parts)
    num_non_numeric = len(parts) - num_numeric
    # Heuristic: if more than half are non-numeric, it's likely a header
    return num_non_numeric > num_numeric


def log_metrics_from_results(file_path:str, default_header:list)-> None:
    """
    Log metrics from a results file to MLflow.
    :param file_path: str, path to the results file
    :param default_header: list, default header to use if the file does not contain a header
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    first_line = lines[0].strip()
    if is_header(first_line):
        header = first_line.split()
        data_lines = lines[1:]
    else:
        header = default_header
        data_lines = lines
    
    for line in data_lines:
        data = line.split()
        epoch = int(data[0].split('/')[0])
        for key, value in zip(header[1:], data[1:]):
            if value.replace(".", "").replace("-", "").isdigit():
                mlflow.log_metric(key, float(value), step=epoch)



def log_params_from_yaml(file_path:str)->None:
    """
    Log parameters from a YAML file to MLflow.
    :param file_path: str, path to the YAML file
    """
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    for key, value in params.items():
        mlflow.log_param(key, value)

def main(base_dir:str,experiment_name:str="TestExperiment1"):
    
    hyp_yaml = os.path.join(base_dir, "hyp.yaml")
    results_file = os.path.join(base_dir, "results.txt")
    opt_yaml = os.path.join(base_dir, "opt.yaml")
    model_path = os.path.join(base_dir, "weights")

    # Start MLflow run
    path_mlflow = "./mlruns"
    mlflow.set_tracking_uri(path_mlflow)
    mlflow.set_experiment(experiment_name)

    mlflow.start_run()
    print(f"MLflow run started for experiment: {experiment_name}")

    # Log hyperparameters from hyp.yaml
    log_params_from_yaml(hyp_yaml)

    # Log metrics from results.txt
    default_header = [
        'Epoch', 'gpu_mem', 'GIoU/box', 'Obj', 'Cls', 'kpt', 'kpt_val', 'total', 'batch', 'img_size', 'P', 'R', 'mAP-50', 'mAP-50-95', 'val_GIoU', 'val_obj', 'val_cls'
    ]
    log_metrics_from_results(results_file, default_header)

    # Log training options from opt.yaml
    log_params_from_yaml(opt_yaml)

    # Log model and YAML files as artifacts
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(hyp_yaml)
    mlflow.log_artifact(results_file)
    mlflow.log_artifact(opt_yaml)

    # End MLflow run
    mlflow.end_run()
    print("MLflow run ended successfully")

if __name__ == "__main__":
    # use  "mlflow ui" in console to start mlflow server 
    #  use ctrl+c to stop cleanly
    # directory with yolov5 trained model with keypoint and boundingbox
    base_dir = "../../number_plate/yolov7-pose/yolov7-pose-custom-changed/training/04_yolov5-plus_ochi_dataset"
    main(base_dir=base_dir, experiment_name ="eperiment1")
