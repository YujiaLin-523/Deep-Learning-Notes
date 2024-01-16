from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


writer = SummaryWriter("logs") # 创建一个logs文件夹，writer写的文件都在该文件夹下
# writer.add_image()
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)
writer.close()

img_path1 = "Dataset/cat_dataset/CAT_00/00000001_000.jpg"
img_PIL1 = Image.open(img_path1)
img_array1 = np.array(img_PIL1)

img_path2 = "Dataset/cat_dataset/CAT_00/00000001_005.jpg"
img_PIL2 = Image.open(img_path2)
img_array2 = np.array(img_PIL2)

writer = SummaryWriter("logs")
writer.add_image("cat", img_array1, 1, dataformats="HWC")
writer.add_image("cat", img_array2, 2, dataformats="HWC")
writer.close()
