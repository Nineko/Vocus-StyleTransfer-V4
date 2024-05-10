import numpy as np
import torch
import os
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
from utils.imgprocess import img_transform_256,img_transform_299,load_image,save_image
from net.transform import TransformNet
from net.stylepredict import StylePredict

content_path = "results/content1.jpg"
style_path = "results/style.png"
model_path = "models/MyStyle.pt"
style_model_path = "models/MyStylePredict.pt"
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_pil = ToPILImage()
print(device)

content = load_image(content_path)
content = img_transform_256(content)
content = content.unsqueeze(0)
content = content.to(device=device, dtype=dtype)
print(content.shape)

style = load_image(style_path)
style = img_transform_299(style)
style = style.unsqueeze(0)
style = style.to(device=device, dtype=dtype)
print(style.shape)

transform_model = TransformNet().to(device=device, dtype=dtype)
transform_model.load_state_dict(torch.load(model_path))
transform_model.eval()

style_model = StylePredict().to(device=device, dtype=dtype)
style_model.load_state_dict(torch.load(style_model_path))
style_model.eval()

print("Ready")
style_vector = style_model(style)
print(style_vector.shape)
stylized = transform_model(content,style_vector)
result_img = stylized[0].cpu().detach()
result_img = result_img.clamp(0, 1)    
img = to_pil(result_img)
img.save(f"results/result.jpg")

print("Over")