print('Импорт библиотек..')
import cv2
import sys, os
import torch
import numpy as np
from omegaconf import OmegaConf
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from effdet import create_model
from effdet import DetBenchPredict
import argparse
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


IMAGE_SIZE = 512

print('Загрузка модели..')

model = create_model(
    'tf_efficientdet_d0',
    bench_task='train',
    num_classes=1,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    bench_labeler=True,
    pretrained=True)


def test_aug():
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ]
    )


def box_transforms(boxes, img_h, img_w):
    boxes = np.hstack(boxes)
    x_i = boxes[0] / IMAGE_SIZE * img_w
    y_i = boxes[1] / IMAGE_SIZE * img_h
    w_i = boxes[2] / IMAGE_SIZE * img_w
    h_i = boxes[3] / IMAGE_SIZE * img_h
    return int(x_i), int(y_i), int(w_i), int(h_i)


def make_predictions(model, images):
    with torch.no_grad():
        images = images.to(device)
        detections = model(images)
        pred = detections[0].detach().cpu().numpy()
        boxes = pred[:, :4]
        scores = pred[:, 4]
        indexes = np.where(scores >= 0.5)[0]
        print(indexes)
        boxes = boxes[indexes]
        print(boxes)
    return boxes


def add_bboxes(img, type='img'):
    img_w = img.shape[1]
    img_h = img.shape[0]
    aug_test = test_aug()
    img_aug = aug_test(image=img)['image']
    img_aug = img_aug.unsqueeze(0)

    boxes = make_predictions(test_net, img_aug)
    if len(boxes) > 0:
        x_i, y_i, w_i, h_i = box_transforms(boxes, img_h, img_w)
        if type == 'img':
            cv2.rectangle(img, (x_i, y_i), (w_i, h_i), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif type == 'video':
            cv2.rectangle(img, (x_i, y_i), (w_i, h_i), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return img


print('Загрузка весов..')

checkpoint_path = './weights/tf_efficientdet_d0_model_with_val_loss_0.2589409126476808.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
test_net = DetBenchPredict(model.model).to(device)

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--source', required=True, help='Путь к картинке или видео')
args = vars(ap.parse_args())

file_path = args['source']
file_name = os.path.basename(file_path)
dir_path = os.path.dirname(file_path)
name, ext = file_name.split('.')
out = '_out.'
new_file_name = name + out + ext
new_file_path = dir_path + '/' + new_file_name

if ext == 'png' or ext == 'jpeg' or ext == 'jpg':
    img = cv2.imread(str(file_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_bboxes(img)
    cv2.imwrite(new_file_path, img[:, :, ::-1])

    print(f'Изображение сохранено в папку "{dir_path}", имя файла "{new_file_name}"')

else:
    try:
        cap = cv2.VideoCapture(file_path)

        while True:
            ret, img = cap.read()
            img = add_bboxes(img, 'video')
            cv2.imshow('video feed', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except:
        print('Тип файла не поддерживается или вы указали неверный путь.')
