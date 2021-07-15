import torch
import torch.nn as nn
from mobilenet_v1 import MobileNet_v1
from torchsummary import summary 

for model_alpha in [0.25, 0.5, 0.75, 1.0]:
  for model_img_size in [128, 160, 192, 224]:
      model_name = "mobilenet_v1_size_" + str(model_img_size) + "_alpha_" + str(model_alpha)
      print("constructing for " + "./pt_model_weights/mobilenet_v1_size_" + str(model_img_size) + "_alpha_" + str(model_alpha) + "_top.pth")
      model = MobileNet_v1(1000, alpha=model_alpha, input_size=model_img_size, include_top=True)
      model.load_state_dict(torch.load("./pt_model_weights/mobilenet_v1_size_" + str(model_img_size) + "_alpha_" + str(model_alpha) + "_top.pth"))
      model.eval()
      one = torch.ones(1, 3, model_img_size, model_img_size)
      out = model(one)
      print(str(out[0].item())+", "+str(out[500].item())+", "+str(out[999].item()))
