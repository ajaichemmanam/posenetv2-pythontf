import tensorflow as tf
import cv2
import time
import argparse
import os

import posenet


MODEL_DIR = './models'
DEBUG_OUTPUT = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model-resnet_v2')
parser.add_argument('--output_stride', type=int, default=16)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

def load_model(model_name, sess, model_dir=MODEL_DIR):
    model_path = os.path.join(model_dir, '%s.pb' % model_name)
    if not os.path.exists(model_path):
        print('Cannot find model file %s' % model_path)

    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    if DEBUG_OUTPUT:
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
            print('Loaded graph node:', t.name)
    #For Mobilenet Version 
    offsets = sess.graph.get_tensor_by_name('MobilenetV1/offset_2/BiasAdd:0')
    displacement_fwd = sess.graph.get_tensor_by_name('MobilenetV1/displacement_fwd_2/BiasAdd:0')
    displacement_bwd = sess.graph.get_tensor_by_name('MobilenetV1/displacement_bwd_2/BiasAdd:0')
    heatmaps = sess.graph.get_tensor_by_name('MobilenetV1/heatmap_2/BiasAdd:0')
    # For Resnet50 Version
    # offsets = sess.graph.get_tensor_by_name('float_short_offsets:0')
    # displacement_fwd = sess.graph.get_tensor_by_name('resnet_v1_50/displacement_fwd_2/BiasAdd:0')
    # displacement_bwd = sess.graph.get_tensor_by_name('resnet_v1_50/displacement_bwd_2/BiasAdd:0')
    # heatmaps = sess.graph.get_tensor_by_name('float_heatmaps:0')

    return [heatmaps, offsets, displacement_fwd, displacement_bwd]


def main():

    with tf.Session() as sess:
        model_outputs = load_model(args.model, sess)
        output_stride = args.output_stride #16 #Change it according to the model

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'sub_2:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
