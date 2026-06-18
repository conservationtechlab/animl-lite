# animl-py 3.2.1
AniML comprises a variety of machine learning tools for analyzing ecological data. This Python package includes a set of functions to classify subjects within camera trap field data and can handle both images and videos. 
This package is also available in R: [animl](https://github.com/conservationtechlab/animl)

Table of Contents
1. Installation
2. [Usage](#usage)
3. [Models](#models)

## Installation Instructions

It is recommended that you set up a conda environment for using animl.
See **Dependencies** below for more detail. You will have to activate the conda environment first each
time you want to run AniML from a new terminal.

### From GitHub
```
git clone https://github.com/conservationtechlab/animl-py.git
cd animl-py
pip install -e .
```

### From PyPi
```
pip install animl
```

### Dependencies
We recommend running AniML on GPU-enabled hardware.
**If using an NVIDIA GPU, ensure driviers, cuda-toolkit and cudnn are installed.

**Python** >= 3.12 <3.14

**Onnx** <br>
Animl currently depends on onnxruntime-gpu == 1.20.0.

Python Package Dependencies
* numpy>=2.0.2
* pandas>=2.2.2,<3.0.0
* pillow>=11.0.0
* opencv-python>=4.12.0.88
* tqdm>=4.66.5


### Verify Install 
We recommend you download the [examples](https://github.com/conservationtechlab/animl-py/blob/main/examples/Southwest.zip) folder within this repository.
Download and unarchive the zip folder. Then with the conda environment active:
```
python -m animl /path/to/example/folder
```
This should create an Animl-Directory subfolder within
the example folder.

Or, if using your own data/models, animl can be given the paths to those files:
Download and unarchive the zip folder. Then with the conda environment active:
```
python -m animl /example/folder --detector /path/to/megadetector --classifier /path/to/classifier --classlist /path/to/classlist.txt
```
You can use animl in this fashion on any image directory.

Finally you can use the animl.yml config file to specify parameters:
```
python -m animl /path/to/animl.yml
```

## Usage

### Inference
The functionality of animl can be parcelated into its individual functions to suit your data and scripting needs.
The sandbox.ipynb notebook has all of these steps available for further exploration.

1. It is recommended that you use the animl working directory for storing intermediate steps.
```python
import animl
workingdir = animl.WorkingDirectory('/path/to/save/data')
```

2. Build the file manifest of your given directory. This will find both images and videos.
```python
files = animl.build_file_manifest('/path/to/images', out_file=workingdir.filemanifest, exif=True)
```

3. If there are videos, extract individual frames for processing.
   Select either the number of frames or fps using the argumments.
   The other option can be set to None or removed.
```python
allframes = animl.extract_frames(files, frames=3, out_file=workingdir.imageframes, parallel=True)
```

4. Pass all images into MegaDetector. We recommend [MDv5a](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt).
   The function parse_MD will convert the json to a pandas DataFrame and merge detections with the original file manifest, if provided.

```python
detector = animl.load_detector('/path/to/mdmodel.pt', model_type="mdv5", device='cuda:0')
mdresults = animl.detect(detector, allframes, resize_width=animl.MEGADETECTORv5_SIZE, resize_height=animl.MEGADETECTORv5_SIZE, 
                         letterbox=True, file_col="frame", device='cuda:0', checkpoint_path=working_dir.mdraw, quiet=True)
detections = animl.parse_detections(mdresults, manifest=allframes, out_file=workingdir.detections)
```

5. For speed and efficiency, extract the empty/human/vehicle detections before classification.
```python
animals = animl.get_animals(detections)
empty = animl.get_empty(detections)
```
6. Classify using the appropriate species model. Merge the output with the rest of the detections
   if desired.
```python
classifier, class_list = animl.load_classifier('/path/to/model', '/path/to/classlist.txt', device='cuda:0')
raw_predictions = animl.classify(classifier, animals, resize_width=480, resize_height=480, 
                                 file_col="filepath", batch_size=4, out_file=working_dir.predictions)
```

7. Apply labels from class list with or without utilizing timestamp-based sequences.
```python
manifest = animl.single_classification(animals, empty, raw_predictions, class_list['class'])

```
or, after defining a station column,
```python
manifest = animl.sequence_classification(animals,
                                         empty, 
                                         raw_predictions,
                                         class_list['class'],
                                         station_col='station',
                                         empty_class="",
                                         sort_columns=None,
                                         file_col="filepath",
                                         maxdiff=60)
```

8. (OPTIONAL) Save the Pandas DataFrame's required columns to csv and then use it to create json for TimeLapse compatibility
```python
csv_loc = animl.export_timelapse(manifest, imagedir, only_animal = True)
animl.export_megadetector(manifest, out_file ="final_result.json", detector = 'MegaDetector v5a')
```

9. (OPTIONAL) Create symlinks within a given directory for file browser access.
```python
manifest = animl.export_folders(manifest, out_dir=working_dir.linkdir, out_file=working_dir.results)
```
