import streamlit as st
import io

# Copyright (c) Facebook, Inc. and its affiliates.
# Library Requirements for Detectron2

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

from predictor import VisualizationDemo

def setup_cfg(confidence_threshold, model):

    if "lowlight_cctv_test" in DatasetCatalog.list():
        DatasetCatalog.remove("lowlight_cctv_test")

    # Get COCO Instances
    register_coco_instances("lowlight_cctv_test", {}, "Data/testset/annotations/instances_default.json", "Data/testset/images")

    cfg = get_cfg()
    cfg.merge_from_file("Faster RCNN Configs/model-config.yaml")
    cfg.merge_from_list(['MODEL.WEIGHTS', "Faster RCNN Configs/" + model + ".pth"])

    cfg.DATASETS.TEST = ("lowlight_cctv_test") # Training Dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # RoIHead Batch Size (Default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Number of Object Classes

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold / 100
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold / 100
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold / 100

    cfg.freeze()
    return cfg

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def processVideo (file):
    
    if file is not None:
        g = io.BytesIO(file.read())  ## BytesIO Object
        temporary_location = "testout_simple.mp4"

        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

        # close file
        out.close()

    video = cv2.VideoCapture(temporary_location)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(file.name)
    codec, file_ext = (
        ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )
    if codec == ".mp4v":
        warnings.warn("x264 codec not available, switching to mp4v")

    output = "test - result.mp4"

    if output:
        if os.path.isdir(output):
            output_fname = os.path.join(output, basename)
            output_fname = os.path.splitext(output_fname)[0] + file_ext
        else:
            output_fname = output

        assert not os.path.isfile(output_fname), output_fname
        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )

    assert os.path.isfile(file.name)
    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        output_file.write(vis_frame)

    video.release()
    output_file.release()



if __name__ == "__main__":

    # Streamlit Data

    st.title("Vehicle Night Detection")

    st.subheader("Configurations")

    option = st.selectbox(
        'Which Model would you like the use?',
        ('All Data', 'All CCTV Data', 'Lowlight Only', 'Daytime Only', 'Nighttime CCTV', 'Daytime NonCCTV', 'Nighttime NonCCTV', 'Daytime CCTV'))
    
    confidence_threshold = st.slider('Confidence Threshold?', 0, 100, 50)

    file = st.file_uploader("Pick a file", type=["mp4"])

    if st.button('Submit'):
        
        mp.set_start_method("spawn", force=True)
        cfg = setup_cfg(confidence_threshold, option)

        demo = VisualizationDemo(cfg)
        processVideo(file)

        st.subheader('Output')

        video_file = open('test - result.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

    else:
        st.write("Please upload a file.")



