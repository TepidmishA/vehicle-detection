"""
Object Detection Module

Provides abstract detection interface and concrete implementations for:
- Production-ready vehicle detection (placeholder)
- Randomized fake detection for testing/development

Classes:
    :Detector: Abstract base class defining detection interface
    :VehicleDetector: For real vehicle detection
    :FakeDetector: Test implementation with random bounding box generation

Dependencies:
    :OpenCV (cv2): for image handling
    :random: for synthetic detection generation
    :abc: for abstract base class support
"""
from urllib.parse import urlparse
import time
from pathlib import Path
from abc import ABC, abstractmethod
import random
from functools import lru_cache
import requests

import cv2 as cv
import numpy as np
import torch
from torchvision.models import detection
from ultralytics import YOLO, RTDETR

import src.vehicle_detector.adapter as ad


class Detector(ABC):
    """
    Abstract base class for object detection implementations.
    
    Defines interface for detection systems using factory pattern.
    
    Methods:
        detect: Abstract detection method
        create: Factory method for instantiating concrete detectors
    
    Supported Modes:
        'vehicle': Production detector
        'fake': Testing detector with random boxes
    """

    def __init__(self, param_detect, adapter):
        self.scale = param_detect['scale']
        self.size = param_detect['size']
        self.mean = param_detect['mean']
        self.swap_rb = param_detect['swapRB']
        self.batch_size = param_detect['batch_size']
        self.adapter = adapter

    @abstractmethod
    def detect(self, images: list[np.ndarray]):
        """
        Process image and return detected objects.

        :param images: Input array of images (OpenCV format)
        """

    @staticmethod
    @lru_cache(maxsize=16)
    def load_classes(path: str) -> tuple:
        """
        Load class labels from a local file or URL with caching.
        """
        parsed = urlparse(path)
        if parsed.scheme in ('http', 'https'):
            response = requests.get(path, timeout=5)
            if response.status_code == 200:
                print(f"[INFO] Loaded class names from URL (cached): {path}")
                return tuple(response.text.strip().split('\n'))

            raise ValueError(f"Failed to load class file from URL: "
                             f"{path} (status code {response.status_code})")

        path = Path(path).absolute()
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return tuple(f.read().strip().split('\n'))

        raise ValueError(f"Incorrect path to class file: {path}")

    @staticmethod
    def create(adapter_name, path_classes, paths, param_adapter, param_detect):
        """
        Factory method for creating detector instances.

        :return: Detector: Concrete subclass instance
        :raise: ValueError: For unsupported mode specifications
        """

        def make_adapter(adapter_class):
            return adapter_class(param_adapter['confidence'],
                                 param_adapter['nms_threshold'],
                                 Detector.load_classes(path_classes))

        config = {
            'AdapterYOLO': lambda:
                VehicleDetectorOpenCV('ONNX', paths, param_detect,
                                      make_adapter(ad.AdapterYOLO)),
            'AdapterYOLOTiny': lambda:
                VehicleDetectorOpenCV('ONNX', paths, param_detect,
                                      make_adapter(ad.AdapterYOLOTiny)),
            'AdapterDetectionTask': lambda:
                VehicleDetectorOpenCV('TensorFlow', paths, param_detect,
                                      make_adapter(ad.AdapterDetectionTask)),
            'AdapterTorchvision': lambda:
                VehicleDetectorTorchvision(paths, param_detect,
                                           make_adapter(ad.AdapterTorchvision)),
            'AdapterUltralytics': lambda:
                VehicleDetectorUltralytics(paths, param_detect,
                                           make_adapter(ad.AdapterUltralytics)),
            'AdapterUltralyticsYoloONNX': lambda:
                VehicleDetectorOpenCV('ONNX', paths, param_detect,
                                      make_adapter(ad.AdapterUltralyticsYoloONNX)),
            'fake': lambda:
                FakeDetector()
        }

        if adapter_name not in config:
            raise ValueError(f"Unsupported adapter: {adapter_name}")

        return config[adapter_name]()


# need testing of batch processing implementation
class VehicleDetectorOpenCV(Detector):
    """
    A class for performing vehicle detection using OpenCV's deep learning module.
    """

    def __init__(self, format_load, paths, param_detect, adapter):

        super().__init__(param_detect, adapter)
        if format_load == 'TensorFlow':
            self.model = cv.dnn.readNetFromTensorflow(paths['path_weights'], paths['path_config'])
        elif format_load == 'Darknet':
            self.model = cv.dnn.readNetFromDarknet(paths['path_config'], paths['path_weights'])
        elif format_load == 'ONNX':
            self.model = cv.dnn.readNetFromONNX(paths['path_weights'])
        else:
            raise ValueError('Incorrect format load.')

    def detect(self, images: list[np.ndarray]):
        orig_image_count = len(images)
        if orig_image_count < self.batch_size:
            for _ in range(self.batch_size - orig_image_count):
                images.append(images[0])

        # Pre-process image using the adapter
        start_time = time.time()
        image_transformed = self.adapter.pre_processing(images,
                                                        scalefactor=self.scale,
                                                        size=self.size,
                                                        mean=self.mean,
                                                        swapRB=self.swap_rb)
        blobs = cv.dnn.blobFromImages(image_transformed)
        preproc_time = time.time() - start_time

        # Perform inference
        start_time = time.time()
        self.model.setInput(blobs)
        outputs = self.model.forward()
        inference_time = time.time() - start_time

        # Post-process the detections using the adapter
        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]  # (width, height)
        detections = self.adapter.post_processing(outputs, image_sizes,
                                                  size=self.size)
        postproc_time = time.time() - start_time

        detections = detections[:orig_image_count]

        return detections, preproc_time, inference_time, postproc_time


class VehicleDetectorTorchvision(Detector, ABC):
    """
    A class for performing vehicle detection using Torchvision models.
    """

    def __init__(self, paths, param_detect, adapter):
        super().__init__(param_detect, adapter)
        if paths['path_weights'] == 'FasterRCNN_ResNet50_FPN_Weights.COCO_V1':
            self.model = detection.fasterrcnn_resnet50_fpn(
                weights=detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        elif paths['path_weights'] == 'FCOS_ResNet50_FPN_Weights.COCO_V1':
            self.model = detection.fcos_resnet50_fpn(
                weights=detection.FCOS_ResNet50_FPN_Weights.COCO_V1)
        elif paths['path_weights'] == 'RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1':
            self.model = detection.retinanet_resnet50_fpn_v2(
                weights=detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        elif paths['path_weights'] == 'SSD300_VGG16_Weights.COCO_V1':
            self.model = detection.ssd300_vgg16(
                weights=detection.SSD300_VGG16_Weights.COCO_V1)
        elif paths['path_weights'] == 'SSDLite320_MobileNet_V3_Large_Weights.COCO_V1':
            self.model = detection.ssdlite320_mobilenet_v3_large(
                weights=detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
        else:
            raise ValueError('Incorrect path to weights in VehicleDetectorTorchvision.')

        self.model.eval()

    def detect(self, images: list[np.ndarray]):
        """
        Performs object detection on the input image.

        :param images: Input list of images (image is a NumPy array).
        :return: List of detections in the format [class, x1, y1, x2, y2, confidence].
        """
        # Pre-process
        start_time = time.time()
        image_tensors = self.adapter.pre_processing(images)
        preproc_time = time.time() - start_time

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image_tensors)
        inference_time = time.time() - start_time

        # post-processing
        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]  # (width, height)
        detections = self.adapter.post_processing(outputs, image_sizes)
        postproc_time = time.time() - start_time

        return detections, preproc_time, inference_time, postproc_time


class VehicleDetectorUltralytics(Detector):
    """
    Vehicle detector based on YOLO and RTDETR ONNX using the Ultralytics API.
    """

    def __init__(self, paths, param_detect, adapter):
        """
        :param param_detect: Dictionary with detection parameters.
        :param adapter: Adapter for pre/post-processing.
        """
        super().__init__(param_detect, adapter)
        if 'yolo' in paths['path_weights']:
            self.model = YOLO(paths['path_weights'])
        elif 'rtdetr' in paths['path_weights']:
            self.model = RTDETR(paths['path_weights'])
        else:
            raise ValueError("VehicleDetectorUltralytics invalid path_weights")

    def detect(self, images: list[np.ndarray]):
        """
        Performs object detection on the input images using YOLOv8.

        :param images: List of input images (BGR numpy arrays).
        :return: Tuple (detections, preproc_time, inference_time, postproc_time)
        """
        # Pre-processing
        start_time = time.time()
        preprocessed = self.adapter.pre_processing(images,
                                                  size=self.size)
        preproc_time = time.time() - start_time

        # Inference
        start_time = time.time()
        results = self.model(preprocessed)
        inference_time = time.time() - start_time

        # Post-processing
        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]
        detections = self.adapter.post_processing(results, image_sizes)
        postproc_time = time.time() - start_time

        return detections, preproc_time, inference_time, postproc_time


class FakeDetector(Detector):
    """
    Testing detector generating random bounding boxes.

    Generates 0-7 random vehicle boxes per image with:
    - Random vehicle classes (car/bus/truck)
    - Random valid positions within image bounds
    - Reproducible results via seed control
    """

    def __init__(self,
                 seed: int = None,
                 time_ranges: tuple[tuple[float, float]] = ((0.01, 0.05),
                                                            (0.08, 0.3),
                                                            (0.01, 0.1))):
        """
        Initialize fake detector with controlled randomness.

        :param seed: Random seed for reproducibility
        :param time_ranges: Tuple of (preproc, inference, postproc) time ranges
        """
        self.rand = random.Random(seed)
        self.time_ranges = time_ranges
        self.classes = ["car", "bus", "truck"]
        super().__init__(None, None)

    def detect(self, images: list[np.ndarray]):
        """
        Process batch of images with simulated detection pipeline.

        :param images: List of numpy arrays (HWC images)
        :return: Tuple containing:
            - List of detections per image
            - Preprocessing time
            - Inference time
            - Postprocessing time
        """
        preproc = self.rand.uniform(*self.time_ranges[0])
        inference = self.rand.uniform(*self.time_ranges[1])
        postproc = self.rand.uniform(*self.time_ranges[2])

        batch_detections = []
        for frame in images[0]:
            if frame is None or frame.size == 0:
                batch_detections.append([])
                continue

            height, width = frame.shape[:2]
            detections = []

            if self.rand.random() < 0.3:
                batch_detections.append([])
                continue

            for _ in range(self.rand.randint(0, 5)):
                try:
                    x1 = self.rand.randint(0, width - 2)
                    x2 = self.rand.randint(x1 + 1, width - 1)
                    y1 = self.rand.randint(0, height - 2)
                    y2 = self.rand.randint(y1 + 1, height - 1)
                except ValueError:
                    continue

                detections.append((
                    self.rand.choice(self.classes),
                    x1, y1, x2, y2,
                    round(self.rand.uniform(0.4, 0.99), 2)
                ))

            batch_detections.append(detections)

        return (
            batch_detections,
            preproc,
            inference,
            postproc
        )
