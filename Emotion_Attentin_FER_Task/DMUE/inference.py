import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.make_target_model import make_target_model
import pandas as pd

class Config:
    pass
cfg = Config()
cfg.ori_shape = (169, 169)
cfg.image_crop_size = (224, 224)
cfg.normalize_mean = [0.5, 0.5, 0.5]
cfg.normalize_std = [0.5, 0.5, 0.5]
cfg.last_stride = 2
cfg.num_classes = 8
cfg.num_branches = cfg.num_classes + 1
cfg.backbone = 'resnet18' # 'resnet18', 'resnet50_ibn'
cfg.pretrained = "./weights/AffectNet_res18_acc0.6285.pth"
cfg.pretrained_choice = '' # '' or 'convert'
cfg.bnneck = True  
cfg.BiasInCls = False

# DMUE 모델로 감정상태 분류 inference 
def inference(model, img_path, transform, is_cuda=True):
    img = Image.open(img_path).convert('RGB')    
    img_tensor = transform(img).unsqueeze(0)
    if is_cuda:
        img_tensor = img_tensor.cuda()
    
    model.eval()
    if is_cuda:
        model = model.cuda()

    pred = model(img_tensor)
    prob = F.softmax(pred, dim=-1)
    idx  = torch.argmax(prob.cpu()).item()
    key = {0: '2', 1: '3', 2: '2', 3: '3', 4:'2', 5:'1', 6:'1', 7:'1'}

    return (os.path.basename(img_path), key[idx])
if __name__ == '__main__':
    path = '/home/ubuntu/ksy/SNU/dataset/deep_face_crop'
    img_path = os.listdir(path)
    results = []
    transform = T.Compose([
        # T.Resize(cfg.ori_shape),
        # T.CenterCrop(cfg.image_crop_size),
        T.ToTensor(),
        # T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])

    # print('Building model......')
    model = make_target_model(cfg)
    model.load_param(cfg)
    for img in img_path:
        keys = inference(model, os.path.join(path,img), transform, is_cuda=True)
        results.append(keys)
    dataframe = pd.DataFrame(results)
    # 결과 저장
    dataframe.to_csv("./deep_face_noalin_sad2_norm_results.csv",
                     header=False, index=False)
