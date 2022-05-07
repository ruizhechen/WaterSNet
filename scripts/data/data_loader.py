# data loader
from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
import random
from torchvision import transforms
import glob
import torch
def transform(train_size):
	return transforms.Compose([
		transforms.Resize((train_size, train_size)),
		transforms.RandomAffine(30),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomGrayscale(p=0.2),
		transforms.ToTensor()
	])

def transform_test(test_size):
	return transforms.Compose([
		transforms.Resize((test_size, test_size)),
		transforms.ToTensor()
	])

class TrainDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list,train_size):
		super(TrainDataset, self).__init__()
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.train_size=train_size
		self.transform = transform(self.train_size)

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		image = Image.open(self.image_name_list[idx]).convert('RGB')
		label = Image.open(self.label_name_list[idx]).convert("L")
		seed = np.random.randint(12345)
		if self.transform:
			random.seed(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			image = self.transform(image)
			random.seed(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			label = self.transform(label)
		return image, label


class TestDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list,test_size):
		super(TestDataset, self).__init__()
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.test_size=test_size
		self.transform = transform_test(self.test_size)

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):

		image = Image.open(self.image_name_list[idx]).convert('RGB')
		label = Image.open(self.label_name_list[idx]).convert("L")
		if self.transform:
			image = self.transform(image)
			label = self.transform(label)

		return image, label

def train_loader(tra_image_list,tra_lbl_list,batch_size_train,train_img_size):
	train_dataset = TrainDataset(img_name_list=tra_image_list,lbl_name_list=tra_lbl_list,train_size=train_img_size)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8,drop_last=True)
	return train_dataloader

def test_loader(test_image_list,batch_size_val,test_img_size):
	test_dataset = TestDataset(img_name_list=test_image_list,lbl_name_list=test_image_list,test_size=test_img_size)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, num_workers=8,drop_last=False)
	return test_dataloader
