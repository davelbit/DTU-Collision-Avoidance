#!/usr/bin/env python3
######################################################################
# Authors:      - Varun Ghatrazu <s210245>
#               - David Parham <s202385>
#
# Course:       Deep learning approaches for damage limitation in car-human collisions
# Semester:     Fall 2021
# Institution:  Technical University of Denmark (DTU)
######################################################################
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Input, LeakyReLU, ReLU, UpSampling2D,
                                     ZeroPadding2D)
from tensorflow.keras.utils import get_custom_objects


def ALReLU(x):
    alpha = 0.01
    return K.maximum(K.abs(alpha*x), x)


get_custom_objects().update({'ALReLU': tf.keras.layers.Activation(ALReLU)})


def parse_cfg(cfg_file):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    with open(cfg_file, "r") as file:
        lines = [line.rstrip("\n") for line in file if line != "\n" and line[0] != "#"]

    holder = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            line = "type=" + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}

        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()

    blocks.append(holder)
    return blocks


def YOLOv3Net(cfg_file, model_size, num_classes):
    """This function creates the Yolov3 model.

    Parameters
    ----------
    cfg_file : [string]
        Path to the yolov3.cfg file.
    model_size : [tuple]

    num_classes : [int]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    blocks = parse_cfg(cfg_file)

    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    inputs = inputs / 255.00

    for i, block in enumerate(blocks[1:]):
        # Darknet-53 model layers
        if block["type"] == "convolutional":
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            activation = block["activation"]

            # downsampling
            if strides > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding="valid" if strides > 1 else "same",
                name="conv_" + str(i),
                use_bias="batch_normalize" not in block,
            )(inputs)

            if "batch_normalize" in block:
                inputs = BatchNormalization(name="bnorm_" + str(i))(inputs)

            if activation == "leaky":
                # inputs = Activation(ALReLU, name="alrelu_" + str(i))(inputs)
                inputs = LeakyReLU(alpha=0.1, name="leaky_" + str(i))(inputs)
                # inputs = ReLU(name="leaky_" + str(i))(inputs)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            inputs = UpSampling2D(stride)(inputs)

        elif block["type"] == "route":
            block["layers"] = block["layers"].split(",")
            start = int(block["layers"][0])

            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)

            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]

        # Yolo detection layer
        elif block["type"] == "yolo":

            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list()

            inputs = tf.reshape(
                inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes]
            )

            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5 : num_classes + 5]

            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (
                input_image.shape[1] // out_shape[1],
                input_image.shape[2] // out_shape[2],
            )
            box_centers = (box_centers + cxy) * strides

            prediction = tf.concat(
                [box_centers, box_shapes, confidence, classes], axis=-1
            )

            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1

        outputs[i] = inputs
        output_filters.append(filters)

    model = Model(input_image, out_pred)
    model.summary()
    return model


if __name__ == "__main__":
    try:
        # main()
        conf = "cfg/yolov3.cfg"
        YOLOv3Net(conf)
    except SystemExit:
        print("STOP")
