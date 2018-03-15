# YOLOV2 on Caffe
This is an implementation of YOLO v2 inferenced using Caffe (pyCaffe) framework

## What's inside
- Yolo to Caffe model converter (thanks to [Duangenquan](https://github.com/duangenquan/YoloV2NCS))

- Sample application of YOLO v2 on PyCaffe (single image) -- `caffe-yolov2/yolo_main.py` (upgraded version of yolo main from [Xingwangsfu](https://github.com/xingwangsfu/caffe-yolo))

- Validation, e.g. 
..* subprocess script for running the main script several times 
..* mAP calculation using VOC dataset (thanks to [AlexeyAB](https://github.com/AlexeyAB/darknet))

## Supported models
Tiny YOLO v2 [cfg](https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg) [weights](https://pjreddie.com/media/files/tiny-yolo-voc.weights)

## How to use it:
**_NOTE !!!_** The following instructions assume that you already have a running Caffe distribution in Python. 

Caffe v1.0 
Installation instruction [here](http://caffe.berkeleyvision.org/installation.html)

### Single image inference 
1. Prepare the config file and pre-trained weights of the model (e.g tiny-yolo-voc)
2. Convert it to Caffe representations(`.prototxt` and `.caffemodel`) using the provided script
3. Run the `yolo_main.py` (add -h for arguments instructions) 

### Validating the performance
1. As an example, we will be using *2012_val.txt* from VOC Dataset as our validation sets.
2. Run ```caffe_valid_run.py``` to run our YOLO parser in Caffe, `yolo_main.py`, against the validation sets.
3. It will produce the needed format in folder `results`

---
## Credits
This application uses Open Source components. You can find the source code of their open source projects below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: Caffe-YOLO by Xingwangsfu (https://github.com/xingwangsfu/caffe-yolo)
for the main application of YOLO v1 using pyCaffe

Project: Darknet by AlexeyAB (originally from pjreddie) https://github.com/AlexeyAB/darknet
for the mAP calculation on PascalVOC

Project: YoloV2NCS by duangenquan https://github.com/duangenquan/YoloV2NCS
for the YOLO v2 output parser and region parameter implementation


