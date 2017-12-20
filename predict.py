import os, cv2, pdb, numpy as np, time, json, pandas
from shapely.geometry import MultiPolygon, box
from subprocess import check_output
from faster_rcnn import network
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.faster_rcnn import FasterRCNN as FasterRCNNModel
from faster_rcnn.nms.py_cpu_nms import py_cpu_nms

pandas.options.mode.chained_assignment = None

class FasterRCNN:
    def __init__(self, weights = None):
        if weights is None:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            download_url = 'https://github.com/ArnholdInstitute/ColdSpots/releases/download/1.0/faster-rcnn.zip'
            if not os.path.exists('weights/faster-rcnn'):
                print('Downloading weights for faster-rcnn')
                if not os.path.exists(os.path.join('weights/faster-rcnn.zip')):
                    check_output(['wget', download_url, '-O', 'weights/faster-rcnn.zip'])
                print('Unzipping...')
                check_output(['unzip', 'weights/faster-rcnn.zip', '-d', 'weights'])
            description = json.load(open('weights/faster-rcnn/description.json'))
            weights = os.path.join('weights/faster-rcnn', description['weights'])
            print('Building model...')
            

        self.model = FasterRCNNModel(classes=['__backround__', 'building'], debug=False)
        network.load_net(weights, self.model)

        self.model.cuda()
        self.model.eval()  


    def close_session(self):
        pass

    def predict_image(self, image, threshold, eval_mode = False):
        """
        Infer buildings for a single image.
        Inputs:
            image :: n x m x 3 ndarray - Should be in RGB format
        """

        if type(image) is str:
            image = cv2.imread(image)
        else:
            image = image[:,:,(2,1,0)] # RGB -> BGR

        im_data, im_scales = self.model.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        t0 = time.time()
        cls_prob, bbox_pred, rois = self.model(im_data, im_info)
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

        current = np.concatenate([
            pred_boxes[:, 4:8], # (skip the background class)
            np.expand_dims(scores[:, 1], 1)
        ], axis=1)

        suppressed = current[py_cpu_nms(current.astype(np.float32), 0.3)]
        suppressed = pandas.DataFrame(suppressed, columns=['x1', 'y1', 'x2', 'y2', 'score'])
        if eval_mode:
            return suppressed[suppressed['score'] >= threshold], suppressed, runtime
        else:
            return suppressed[suppressed['score'] >= threshold]

    def predict_all(self, test_boxes_file, threshold, data_dir = None):
        test_boxes = json.load(open(test_boxes_file))
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(test_boxes_file))
        
        total_time = 0.0

        for i, anno in enumerate(test_boxes):
            orig_img = cv2.imread('%s/%s' % (data_dir, anno['image_path']))[:,:,(2,1,0)]

            pred, all_rects, time = self.predict_image(orig_img, threshold, eval_mode = True)

            pred['image_id'] = i
            all_rects['image_id'] = i

            yield pred, all_rects, test_boxes[i]







