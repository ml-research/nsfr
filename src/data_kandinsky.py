import os
import cv2
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np


class KANDINSKY(torch.utils.data.Dataset):
    """Kandinsky Patterns dataset.
    """

    def __init__(self, dataset, split, img_size=128):
        self.img_size = img_size
        assert split in {
            "train",
            "val",
            "test",
        }
        self.image_paths, self.labels = load_images_and_labels(
            dataset=dataset, split=split)

    def __getitem__(self, item):
        image = load_image_yolo(
            self.image_paths[item], img_size=self.img_size)
        image = torch.from_numpy(image).type(torch.float32) / 255.

        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.labels)


def load_images_and_labels(dataset='twopairs', split='train', img_size=128):
    """Load image paths and labels for kandinsky dataset.
    """
    image_paths = []
    labels = []
    folder = 'data/kandinsky/' + dataset + '/' + split + '/'
    true_folder = folder + 'true/'
    false_folder = folder + 'false/'

    filenames = sorted(os.listdir(true_folder))[:5000]
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(true_folder, filename))
            labels.append(1)

    filenames = sorted(os.listdir(false_folder))[:5000]
    for filename in filenames:
        if filename != '.DS_Store':
            image_paths.append(os.path.join(false_folder, filename))
            labels.append(0)
    return image_paths, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """A utilitiy function for yolov5 model to make predictions. The implementation is from the yolov5 repository.
    """
    import cv2

    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def load_image_yolo(path, img_size, stride=32):
    """Load an image using given path.
    """
    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, 'Image Not Found ' + path
    img = cv2.resize(img0, (img_size, img_size))

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    return img
