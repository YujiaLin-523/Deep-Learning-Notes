from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


img_path = "Dataset/cat_dataset/CAT_00/00000001_000.jpg"
img = Image.open(img_path)

trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
# print(img_tensor)

writer = SummaryWriter("logs")
writer.add_image("img_tensor", img_tensor)
writer.close()

# Normalize归一化
print(img_tensor[0][0][0])
# input[channel]=(input[channel]-mean[channel])/std[channel]
tensor_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = tensor_norm(img_tensor)  
print(img_norm[0][0][0])

writer.add_image("img_tensor", img_tensor)
writer.add_image("img_norm", img_norm)
writer.close()

# Resize拉伸至指定大小
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)

writer.add_image("img_resize", img_resize)
writer.add_image("img_tensor", img_tensor)

# Resize等比缩放
trans_resize_2 = transforms.Resize(600)
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])
img_compose = trans_compose(img)

writer.add_image("img_compose", img_compose)
writer.add_image("img_tensor", img_tensor)
writer.close()

# RandomCrop随机裁剪
trans_random = transforms.RandomCrop(300)  # 可以用(x, y)代替括号中的参数来指定随即裁剪的宽和高
trans_compose_2 = transforms.Compose([trans_random, trans_tensor])

for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
writer.close()
