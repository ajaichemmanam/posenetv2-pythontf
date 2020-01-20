# posenetv2-pythontf
This is a Python and Tensorflow implementation of Posenet v2 released by Google in TensorflowJS.

## Getting Started
1. Clone this repo
2. Add images to input folder
3. Run the main.py as following

```
python3 main.py --model model-mobilenet_v2 --output_stride=16 --image_dir ./images --output_dir ./output
```

## SAMPLE OUTPUT 
![Input](https://raw.githubusercontent.com/ajaichemmanam/posenetv2-pythontf/master/images/0002_c3s1_068642_02.jpg)
![Output](https://raw.githubusercontent.com/ajaichemmanam/posenetv2-pythontf/master/output/0002_c3s1_068642_02.jpg)

## Other Variants of Posenet TFJS Models
1. Download models and weights from the links given in TFJS Model URL
2. Install Tensorflow (tested on ver 1.15.0) and TensorflowJS (tested on ver 1.4.0)
3. Install tfjs-to-tf converter from https://github.com/ajaichemmanam/tfjs-to-tf (CREDITS: @patlevin)
4. After Installation run `tfjs_graph_converter path/to/js/model path/to/frozen/model.pb` in terminal
5. Copy the converted model to models folder 
6. Run

```
python3 main.py --model MODELNAME --output_stride=OUTPUTSTRIDE --image_dir ./images --output_dir ./output
```

## Credits
1. Patrick Levin for TFJS to TF converter
2. Ross Wightman for initial work and code base (https://github.com/rwightman/posenet-python) on posenet