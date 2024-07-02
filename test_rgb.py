# test phase
import torch
import clip
from PIL import Image
import os
from torch.autograd import Variable
from Net import  Net_G
import utils
import numpy as np
import torch.nn.functional as F
import time
import numpy as np    
import cv2    
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    test_path = "/"

    in_c = 2
    out_c = 1
    model_path = ""

    with torch.no_grad():

        model = load_model(model_path, in_c, out_c, device)
        output_path = '/'   
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)
        for i in range(250):

            index = i+1
            infrared_path = test_path + f'ir/{index}.png'
            visible_path = test_path + f'vis/{index}.png'
            text_path = test_path + f"/text/{index}_5.txt"
            with open(text_path, 'r') as f:
                description = f.readline().strip()
            # description = "This is the infrared and visible image fusion task."
            # description = ""
            run_demo(device, clip_model, model, infrared_path, visible_path, description, output_path, index)
    print('Done......')


def load_model(path, input_nc, output_nc, device):

    TextFusionNet_model = Net_G()

    TextFusionNet_model.load_state_dict(torch.load(path, map_location=device))
    TextFusionNet_model.to(device)
    # TextFusionNet_model.load_state_dict(torch.load(path))
    # TextFusionNet_model = torch.nn.DataParallel(TextFusionNet_model,device_ids=[1]);

    para = sum([np.prod(list(p.size())) for p in TextFusionNet_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(TextFusionNet_model._get_name(), para / 1000/1000))
    # print("Params(M): %.3f" % (params_count(TextFusionNet_model) / (1000 ** 2)))

    TextFusionNet_model.eval()

    return TextFusionNet_model
    

def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]
    
    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel

def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])
    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image



def run_demo(device, clip_model, model, infrared_path, visible_path, description, output_path_root, index):

    ir_img = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)
    vi_img = Image.open(visible_path).convert("RGB")
    H, W = ir_img.shape

    # 调整图像尺寸为偶数
    h, w = ir_img.shape
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    ir_img = cv2.resize(ir_img, (new_w, new_h))
    vi_img = vi_img.resize((new_w, new_h))

    vi_img_y, vi_img_cb, vi_img_cr = rgb_to_ycbcr(vi_img)
    
    text = clip.tokenize([description]).to(device)
    description_features = clip_model.encode_text(text)
    
    ir_img = ir_img / 255.0
    vi_img = vi_img_y / 255.0
    
    h = vi_img.shape[0]
    w = vi_img.shape[1]
    
    ir_img_patches = np.resize(ir_img, [1, 1, h, w])
    vi_img_patches = np.resize(vi_img, [1, 1, h, w])
    
    ir_img_patches = torch.from_numpy(ir_img_patches).float()
    vi_img_patches = torch.from_numpy(vi_img_patches).float()
    
    
    ir_img_patches = ir_img_patches.cuda(device)
    vi_img_patches = vi_img_patches.cuda(device)
    model = model.cuda(device)
    
    output, _, _ = model(vis=vi_img_patches, ir=ir_img_patches, text_features=description_features)
    fuseImage = np.zeros((h, w))
    
    out = output.cpu().numpy()
    
    fuseImage = out[0][0]
    
    fuseImage = fuseImage * 255
    
    fuseImage = ycbcr_to_rgb(fuseImage, vi_img_cb, vi_img_cr)
    fuseImage = fuseImage.resize((W,H))
    
    file_name = f'{index}.png'
    if os.path.exists(output_path_root) is False:
        os.mkdir(output_path_root)
    output_path = os.path.join(output_path_root, file_name)

    fuseImage.save(output_path)

    print(output_path)



if __name__ == '__main__':
    main()
