import os
import torch
from torch.autograd import Variable
from configs.config import get_gonfig
import numpy as np
from PIL import Image
import glob
from data.data_loader import test_loader
from models.WaterSNet import WaterSNet
from metrics import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	args=get_gonfig('../config.yaml')
	device=torch.device(args['Test']['device'] if torch.cuda.is_available() else "cpu")
	test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.jpg')
	testset_name=args['Test']['testset_name']
	prediction_dir = '../eval_results/'+testset_name+'/'
	gt_dir=args['Test']['test_mask_dir']
	if not os.path.exists(prediction_dir):
		os.makedirs(prediction_dir)	
	net_dir = args['Test']['ckpt']
	ckpt=torch.load(net_dir)
	# --------- 2. dataloader ---------
	test_loader=test_loader(test_img_name_list,args['Test']['batch_size'],args['Test']['test_size'])
	# --------- 3. load model ---------
	print("...load Net...")
	net=WaterSNet()
	net.load_state_dict(ckpt['net'])	
	net.to(device)
	net.eval()	
	# --------- 4. inference for each image and calculate metrics---------
	for i_test, data_test in enumerate(test_loader):	
		print("inferencing:",test_img_name_list[i_test].split("/")[-1])
		inputs_test = data_test[0]
		inputs_test = inputs_test.type(torch.FloatTensor)
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.to(device))
		else:
			inputs_test = Variable(inputs_test)
		d0= net(inputs_test)
		# normalization
		pred = d0[:, 0, :, :]
		pred = normPRED(pred)
		pred = pred.squeeze().cpu().detach().numpy()
		mask = Image.fromarray((pred * 255).astype(np.uint8))
		name = test_img_name_list[i_test].split("/")[-1]
		name = name.split(".")[0]
		mask.save(prediction_dir+'{}'.format(name+'.png'))
	metrics(prediction_dir,gt_dir)
