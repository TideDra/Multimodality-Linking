import torch
import torchvision.transforms
from PIL import Image, ImageOps

model = torch.load("models/HWClassification_14.pth")
img = Image.open("inputs/008.png")
img = img.convert("L")
img = ImageOps.invert(img)
img.show()
img = img.resize([32, 32])
print(img.size)
trans = torchvision.transforms.ToTensor()
img = trans(img)
img = img.unsqueeze(0)
img = img.cuda()
model.eval()
output = model(img)
pred = output.argmax(1)
print(pred.item())
