# YOLOV2 - Caffe Converter
This is an implementation of YOLO v2 model to Caffemodel converter, thanks to [Duangenquan](https://github.com/duangenquan/YoloV2NCS)

## Supported models
Tiny YOLO v2

Anchor box (Region layer) is implemented in the parser / applications.
This converter does not support the Route and Reorg layer (yet).

..* Note: **yolov2Tiny20** in the *yolomodels* is actually **tiny-yolo-voc** from YOLO original [web](https://pjreddie.com/darknet/yolo)

## How to use it:
1. Prepare the pre-trained YOLO models (config file *.cfg* and pre-trained weights *.weights*).

2. Convert the config file using *create_yolo_prototxt.py* 
```python create_yolo_prototxt.py <cfg_file> <prototxt_output>```

3. Convert the pre-trained weights using *create_yolo_caffemodel.py*
```python create_yolo_caffemodel.py -m <model_file> -w <yoloweight_file> -o <caffemodel_output>```

### Warning!
Please note that unlike Darknet, Caffe does not support default padding in their pooling layers. Depends on your model, you might find a difference in the output size.
In our case of tiny-yolo-voc, Darknet produces 13x13x125 output whilst Caffe produces 12x12x125.
