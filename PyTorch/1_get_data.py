from torch.utils.data import Dataset
from PIL import Image
import os


# 获取单张图片并展示
# img_path = "cat_dataset/CAT_00/00000001_000.jpg"
# img = Image.open(img_path)
# img.show()


class Dataloader(Dataset):
    def __init__(self, root_dir, label_dir):
        self.label_dir = label_dir
        self.root_dir = root_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "Dataset/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Dataloader(root_dir, ants_label_dir)
bees_dataset = Dataloader(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset
img, label = train_dataset[1]
img.show()
