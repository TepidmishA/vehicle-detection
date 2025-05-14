"""
CLI Application "Vehicle Detector"

A command-line interface application for detecting vehicles in images or video streams. 
Supports multiple visualization modes, result recording, and accuracy calculations against 
groundtruth data.

Key Features:
- Vehicle detection in image/video inputs
- GUI/CLI visualization options
- Result export capabilities
- Precision metrics calculation (TPR, FDR, mAP)
- Flexible pipeline configuration

Modules:
- data_reader: Handles input data loading
- detector: Contains vehicle detection logic
- visualizer: Provides visualization interfaces
- writer: Manages result storage
- accuracy_checker: Computes detection accuracy metrics
"""
from dataclasses import asdict
import json
import sys
from pathlib import Path
import argparse
import config_parser

sys.path.append(str(Path(__file__).parent.parent))

from src.gui_application.visualizer import BaseVisualizer
from src.utils import data_reader as dr
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector
from src.detector_pipeline.detector_pipeline import PipelineComponents, DetectionPipeline
from src.accuracy_checker.accuracy_checker import AccuracyCalculator


def cli_argument_parser():
    """
    Parse command-line arguments for the visualizer application.

    Returns:
        argparse.Namespace: Parsed arguments

    Raises:
        argparse.ArgumentError:
            If required arguments are missing or invalid combinations are provided
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yaml',
                        type=str,
                        help='Path to a yaml file',
                        dest='yaml_file',
                        required=True)

    args = parser.parse_args()
    return args


def config_main(parameters: dict):
    """
    Configure pipeline components.

    :param parameters: Parsed command-line arguments
    :return PipelineComponents: Configured pipeline objects
    """

    if parameters['mode'] == 'image':
        reader = FrameDataReader.create(parameters['mode'], parameters['images_path'],
                                        parameters['batch_size'])
    elif parameters['mode'] == 'video':
        reader = FrameDataReader.create(parameters['mode'], parameters['video_path'],
                                        parameters['batch_size'])
    else:
        raise ValueError(f"Unsupported mode: '{parameters['mode']}'. Expected 'image' or 'video'")

    param_detect = {
        'scale': parameters['scale'],
        'size': parameters['size'],
        'mean': parameters['mean'],
        'swapRB': parameters['swapRB']
    }

    param_adapter = {
        'confidence': parameters['confidence'],
        'nms_threshold': parameters['nms_threshold']
    }

    paths = {
        'path_weights': parameters['path_weights'],
        'path_config': parameters['path_config'],
        'path_anchors': parameters['path_anchors']
    }

    detector = Detector.create(parameters['adapter_name'], parameters['path_classes'],
                               paths, param_adapter, param_detect)

    visualizer = BaseVisualizer.create(parameters['silent_mode'])
    writer = Writer.create(parameters['write_path']) if parameters['write_path'] else None
    gt_reader = dr.CsvGTReader(parameters['groundtruth_path']) \
        if parameters['groundtruth_path'] else None

    return PipelineComponents(
        reader, detector, visualizer, writer, gt_reader
    )


def main():
    """"
    Main execution flow for vehicle detection application.

    Workflow:
    1. Parse command-line arguments
    2. Configure detection pipeline
    3. Run vehicle detection process
    4. Calculate and display accuracy metrics (if groundtruth provided)
    5. Handle and report runtime errors

    Requires:
    - Valid input path matching selected mode
    - Accessible detection model file
    - Compatible groundtruth format (when provided)
    """
    try:
        args = cli_argument_parser()
        parameters = config_parser.parse_yaml_file(args.yaml_file)
        components = config_main(parameters)
        pipeline = DetectionPipeline(components)
        pipeline.run()

        if parameters.get('timings_path'):
            data = asdict(pipeline.batches_timings)
            data['total_images'] = components.reader.get_total_images()
            with open(parameters['timings_path'], 'w', encoding="utf-8") as file:
                json.dump(data, file)

        if all([parameters['groundtruth_path'], parameters['write_path']]):
            accur_calc = AccuracyCalculator()
            accur_calc.load_detections(parameters['write_path'])
            accur_calc.load_groundtruths(parameters['groundtruth_path'])
            print(f"TPR: {accur_calc.calc_tpr()}\n"
                  f"FDR: {accur_calc.calc_fdr()}\n"
                  f"MAP: {accur_calc.calc_map()}")

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
