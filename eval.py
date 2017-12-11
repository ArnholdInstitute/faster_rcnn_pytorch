#!/usr/bin/env python

import os, torch, cv2, cPickle, numpy as np, argparse, pdb, json, pandas, time
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms
from faster_rcnn.nms.py_cpu_nms import py_cpu_nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help='Path to weights to use')
parser.add_argument('--test_boxes', default='../data/val_data.json', help='Path to valdation data')
args = parser.parse_args()

# hyper-parameters
# ------------
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'

rand_seed = 1024

max_per_image = 300
thresh = 0.05
vis = True

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def im_detect(net, image):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)

    t0 = time.time()
    cls_prob, bbox_pred, rois = net(im_data, im_info)
    runtime = time.time() - t0

    scores = cls_prob.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes, runtime

def test_net(net, dataset, max_per_image=300, thresh=0.05, vis=False, data_dir='./'):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    classes = ['__backround__', 'building']
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    
    all_boxes = []

    total_time = 0.0

    for i in range(num_images):
        im = cv2.imread(os.path.join(data_dir, dataset[i]['image_path']))
        _t['im_detect'].tic()
        scores, boxes, current_time = im_detect(net, im)
        total_time += current_time
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if vis:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im)

        # skip j = 0, because it's the background class
        for j in xrange(1, len(classes)):
            current = np.concatenate([
                boxes[:, j*4:(j+1)*4],
                np.expand_dims(scores[:, 1], 1),
                np.ones((len(boxes), 1)) * i
            ], axis=1)

            all_boxes.extend(current[py_cpu_nms(current.astype(np.float32), 0.3)])

            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = py_cpu_nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                im2show = vis_detections(im2show, classes[j], cls_dets)
                cv2.imwrite('samples/image_%d.jpg' % i, im2show)

        nms_time = _t['misc'].toc(average=False)

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, detect_time, nms_time)
    df = pandas.DataFrame(all_boxes)
    df.columns = ['x1', 'y1', 'x2', 'y2', 'score', 'image_id']
    df.to_csv('predictions.csv', index=False)
    print('Total time: %.4f, per image: %.4f' % (total_time, total_time / num_images))


if __name__ == '__main__':

    # load net
    net = FasterRCNN(classes=['__backround__', 'building'], debug=False)
    network.load_net(args.weights, net)
    print('load model successfully!')

    net.cuda()
    net.eval()  

    val_data = json.load(open(args.test_boxes))

    # evaluation
    test_net(net, val_data, max_per_image, thresh=thresh, vis=vis, data_dir='../data')
