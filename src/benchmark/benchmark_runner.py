"""
Experiment Runner Script (CLI-based)

Executes batch detection experiments by invoking `cli_main.py` in silent mode with temporary
YAML configs. Results are collected from output CSVs, performance is measured by timing
the subprocess, and accuracy is computed via AccuracyCalculator. Each CLI call is
configured to utilize all CPU threads.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import argparse
import multiprocessing as mp

import pandas as pd
import yaml
from tqdm import tqdm

from src.detector_pipeline.detector_pipeline import BatchesTimings
from src.accuracy_checker.accuracy_checker import AccuracyCalculator
from src.perf_calculator.perf_calculator import PerformanceCalculator
from src.benchmark.generate_plots import generate_perf_plots, generate_quality_plot
from src.benchmark.config_benchmark import ExperimentParameters


def experiment_argument_parser():
    """
    Parses command-line arguments for the detection experiment.

    :return: Parsed arguments including input data path, groundtruth path, output path, and mode.
    """
    parser = argparse.ArgumentParser(description='Run detection experiments')
    parser.add_argument('-in', '--input_data_path',
                        type=str,
                        help='Path to images or video',
                        required=True)
    parser.add_argument('-gt', '--groundtruth_path',
                        type=str,
                        help='Path to groundtruth',
                        required=True)
    parser.add_argument('-out', '--output_path',
                        type=str,
                        help='Output directory for results',
                        default='./results')
    parser.add_argument('-m', '--mode',
                        type=str,
                        choices=['image', 'video'],
                        default='image')
    return parser.parse_args()


def run_process(tmp_path: str, cfg: list, model_cfg_path: str, batch_size: int):
    """
    Executes a subprocess to run a model experiment using the specified configuration.

    :param tmp_path : Path to the temporary YAML configuration file for the experiment.
    :param cfg: List of dictionaries containing the experiment's model configuration.
    :param model_cfg_path: Path to the YAML file with the model configuration.
    :param batch_size: The batch size for which the experiment is executed.
    """
    env = os.environ.copy()
    n_threads = mp.cpu_count()
    env.update({
        'OMP_NUM_THREADS': str(n_threads),
        'MKL_NUM_THREADS': str(n_threads),
        'NUMEXPR_NUM_THREADS': str(n_threads)
    })
    proc = subprocess.run(
        [sys.executable, str(Path('samples') / 'cli_main.py'), '--yaml', tmp_path],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode('utf-8', errors='ignore')
        model_name = cfg[0].get('model_name', Path(model_cfg_path).stem)
        raise RuntimeError(
            f"Experiment failed for model '{model_name}' (batch_size={batch_size}): {stderr}"
        )


def run_single_experiment(model_cfg_path: str, batch_size: int,
                          params: ExperimentParameters,
                          tmp_dir: Path):
    """
    Runs a single experiment for the given model with the specified batch size.
    :param model_cfg_path: Path to the YAML model configuration file.
    :param batch_size: The batch size to be used in the experiment.
    :param params: Experiment parameters, including paths to data, output files, and other settings.
    :param tmp_dir: Path to the temporary CSV file for storing data.
    :return: A dictionary containing the experiment results:
             model name, batch size, performance metrics, and accuracy.
    """
    # Load base YAML
    with open(model_cfg_path, 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override fields
    cfg[0]['mode'] = params.mode
    cfg[0]['images_path'] = params.input_data_path
    cfg[0]['groundtruth_path'] = params.groundtruth_path
    cfg[0]['batch_size'] = batch_size
    cfg[0]['silent_mode'] = True

    # Prepare output CSV path
    out_csv = tmp_dir / f"{Path(model_cfg_path).stem}_bs{batch_size}.csv"
    timings_json = tmp_dir / f"{Path(model_cfg_path).stem}_bs{batch_size}_timings.json"
    cfg[0]['write_path'] = str(out_csv)
    cfg[0]['timings_path'] = str(timings_json)

    # Write temp YAML
    fd, tmp_path = tempfile.mkstemp(suffix='.yaml', dir=tmp_dir)
    os.close(fd)
    with open(tmp_path, 'w', encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Run subprocess in project root with configured env
    run_process(tmp_path, cfg, model_cfg_path, batch_size)

    # Compute perf metrics
    with open(timings_json, 'r', encoding="utf-8") as jf:
        import_data = json.load(jf)

    perf_metrics = PerformanceCalculator.calculate(
        import_data['total_images'],
        batch_size,
        BatchesTimings(
            preprocess_time=import_data['preprocess_time'],
            inference_time=import_data['inference_time'],
            postprocess_time=import_data['postprocess_time']
        )
    )

    # Compute accuracy
    acc_calc = AccuracyCalculator()
    acc_calc.load_detections(out_csv)
    acc_calc.load_groundtruths(params.groundtruth_path)
    accuracy_map = acc_calc.calc_map()

    return {
        'model': cfg[0].get('model_name'),
        'batch_size': batch_size,
        **perf_metrics,
        'accuracy_map': accuracy_map
    }


def run_experiments(dinamic_model_cfgs: list, onnx_model_cfgs: list,
                    batch_sizes: list,
                    params: ExperimentParameters):
    """
    Executes batch detection experiments with given model configurations and batch sizes.
    Runs detection pipelines, collects performance and accuracy metrics, saves results to CSV,
    and generates analysis plots.
    :param dinamic_model_cfgs: List of paths to dinamic model configuration YAML files.
    :param onnx_model_cfgs: List of paths to onnx model configuration YAML files.
    :param batch_sizes: List of batch sizes to be tested.
    :param params: Experiment parameters including input/output paths and mode.
    """
    os.makedirs(params.output_path, exist_ok=True)

    tmp_dir = Path('./src/benchmark/tmp').absolute()
    shutil.rmtree(tmp_dir, ignore_errors=True)  # Ensure clean tmp directory
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # models with dinamic shape
    for model_cfg in tqdm(dinamic_model_cfgs, desc='Models'):
        for bs in tqdm(batch_sizes, desc='Batch sizes', leave=False):
            res = run_single_experiment(model_cfg, bs, params, tmp_dir)
            results.append(res)

    # onnx models for openCV
    for model_cfg in tqdm(onnx_model_cfgs, desc='Models'):
        for bs in tqdm(batch_sizes, desc='Batch sizes', leave=False):
            # Read original config template
            orig_cfg_path = Path(model_cfg)
            cfg_text = orig_cfg_path.read_text(encoding='utf-8')


            # Write to temporary config file
            tmp_cfg_filename = f"{orig_cfg_path.stem}_bs{bs}.yaml"
            tmp_cfg_path = tmp_dir / tmp_cfg_filename
            tmp_cfg_path.write_text(cfg_text.replace('{batch_size}', str(bs)), encoding='utf-8')

            res = run_single_experiment(str(tmp_cfg_path), bs, params, tmp_dir)
            results.append(res)

    # Save results
    df = pd.DataFrame(results)
    csv_out = Path(params.output_path) / 'benchmark_results.csv'
    df.to_csv(csv_out, index=False)

    # Cleanup temporary files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Generate plots
    generate_perf_plots(df, params.output_path)
    generate_quality_plot(df, params.output_path)


if __name__ == '__main__':
    DEFAULT_DINAMIC_MODEL_CONFIGS = [
        './configs/torchvision/detector_config_fasterRCNN.yaml',
        './configs/torchvision/detector_config_FCOS.yaml',
        './configs/torchvision/detector_config_RetinaNet.yaml',
        './configs/torchvision/detector_config_SSD.yaml',
        './configs/torchvision/detector_config_SSDlite.yaml',

        './configs/yolo/detector_config_yolov3_tinyu.yaml',
        './configs/yolo/detector_config_yolov11s.yaml',
        './configs/yolo/detector_config_yolov12s.yaml',
        './configs/rtdetr/detector_config_rtdetr-l.yaml',
    ]
    DEFAULT_ONNX_MODEL_CONFIGS = [
        './configs/onnx/detector_config_yolov3-tinyu_onnx.yaml',
        './configs/onnx/detector_config_yolov11s_onnx.yaml',
        './configs/onnx/detector_config_yolov12s_onnx.yaml'
    ]

    DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]

    try:
        args = experiment_argument_parser()
        data_params = ExperimentParameters(
            args.input_data_path,
            args.groundtruth_path,
            args.output_path,
            args.mode
        )
        run_experiments(
            DEFAULT_DINAMIC_MODEL_CONFIGS,
            DEFAULT_ONNX_MODEL_CONFIGS,
            DEFAULT_BATCH_SIZES,
            data_params,
        )

    except Exception as e:
        print(e)
