
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch, cv2, boto3
from torch.utils import data
import glob, pdb, os, re, json, random, numpy as np, torch
from shapely.geometry import shape
from datetime import datetime
from skimage import io
from config import cfg
from blob import prep_im_for_blob

class Dataset(data.Dataset):
    def __init__(self, root_dir, samples, transform = lambda a1,a2,a3: (a1,a2,a3), no_blanks = True):
        self.root_dir = root_dir   
        self.transform = transform
        self.samples = samples
        self.no_blanks = no_blanks

    def even(self):
        projs = {}
        for sample in self.samples:
            if len(sample['rects']) > 0:
                proj = re.search('(.*[^\d])(\d+)\.', os.path.basename(sample['image_path'])).group(1)
                if proj in projs:
                    projs[proj].append(sample)
                else:
                    projs[proj] = [sample]

        samples = []

        max_size = max([len(projs[k]) for k in projs.keys()])
        for proj in projs.keys():
            arr = projs[proj]
            count = 0
            while count + len(arr) <= max_size and count / 10 < len(arr):
                samples.extend(arr)
                count += len(arr)
            diff = (max_size - len(arr)) % len(arr)
            print('Added %d samples from %s' % (diff + count, proj))
            random.shuffle(arr)
            samples.extend(arr[:diff])
        self.samples = samples
        return self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['image_path']

        if len(sample['rects']) == 0:
            return self[random.randint(0, len(self) - 1)]

        img_data = cv2.imread(os.path.join(self.root_dir, sample['image_path']))
        if img_data is None:
            pdb.set_trace()
        boxes = []
        for f in sample['rects']:
            boxes.append([f['x1'], f['y1'], f['x2'], f['y2'], 1])

        targets = np.array(boxes).astype(float)

        mask = ((targets[:, 2] - targets[:, 0]) > 3) & ((targets[:, 3] - targets[:, 1]) > 3)
        targets = targets[mask, :].astype(float)

        # convert to relative coordinates
        targets[:, (0, 2)] /= img_data.shape[1]
        targets[:, (1, 3)] /= img_data.shape[0]

        if len(targets) == 0:
            targets = np.zeros((0, 5))


        img_data, im_scale = prep_im_for_blob(img_data, cfg.PIXEL_MEANS, cfg.TRAIN.SCALES[-1], cfg.TRAIN.MAX_SIZE)

        targets[:, (0, 2)] *= img_data.shape[1]
        targets[:, (1, 3)] *= img_data.shape[0]

        input_, targets, labels = self.transform(img_data, targets[:, :4], targets[:, -1])

        if len(targets) == 0 and self.no_blanks:
            return self[random.randint(0, len(self)-1)]

        return (
            input_.astype('float32'),
            np.hstack((targets, np.expand_dims(labels, axis=1))).astype('float32'),
            np.array(list(img_data.shape[:2]) + [im_scale], dtype='float32')
        )



