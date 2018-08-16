MobileNetV2-SSD
===========================
This is a caffe implementation of MobileNet-SSD V2 detection network. It is refered to the model released by tensorflow api [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

You can also refer to this model in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segment](128.84.21.199/abs/1801.04381), it is Mobilenet V2+SSDLite architecture.

All these file are created for pascal voc which has 21 classes(one is background), you can change it to suit your dataset.

Pretrain model
===========================
Caffe model from pascal voc 0712 dataset can be down from(https://pan.baidu.com/s/1Hht2LeFiJsxztrGwU5ZyOg).

test script
===========================
mobile_test_save.py is an test file for only two class, you can modify it to suit your class

To do
===========================
1. an end to end training script for your own data

