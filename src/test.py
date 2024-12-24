import os
import torch
from torchvision.utils import save_image
from utils import *
from options import TestOptions
from models import BUM
from datasets import ImgDataset, ImgLoader
import pickle

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

# Set the arguments as specified
opt.outputs_dir = '/home/linhdang/workspace/binhan_workspace/Image_resotre_GAN/FCL-GAN/output'
opt.experiment = 'experiment'
opt.data_source = '/home/linhdang/workspace/binhan_workspace/Image_resotre_GAN/FCL-GAN/data/blurry/val'
opt.pretrained_dir = '/home/linhdang/workspace/binhan_workspace/Image_resotre_GAN/FCL-GAN/pretrained'
opt.model_name = 'latest.pkl'
opt.save_image = True

single_dir = opt.outputs_dir + '/' + opt.experiment + '/single'
multiple_dir = opt.outputs_dir + '/' + opt.experiment + '/multiple'
clean_dir(single_dir, delete=opt.save_image)
clean_dir(multiple_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('testing data loading...')

# Print the structure of the dataset directory
for root, dirs, files in os.walk(opt.data_source):
    print(root, dirs, files)

test_dataset = ImgDataset(data_source=opt.data_source, mode='val')
test_dataloader = ImgLoader(test_dataset, batch_size=1, num_workers=1)
print('Successfully loading validating pairs. =====> qty:{}'.format(len(test_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
model = BUM().cuda()

checkpoint_path = opt.pretrained_dir + '/' + opt.model_name
checkpoint = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(checkpoint['model'])

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')

def main():
    model.eval()
    
    psnr_meter = AverageMeter()
    
    for i, (imgs, gts) in enumerate(test_dataloader):
        real_b, real_s = imgs.cuda(), gts.cuda()
        
        # Print the shape of the loaded images
        print(f'Batch {i}: real_b shape: {real_b.shape}, real_s shape: {real_s.shape}')
        
        with torch.no_grad():
            fake_s = model.forward_G_B2S(real_b)

        cur_psnr = get_metrics(fake_s, real_s) / fake_s.shape[0]
        psnr_meter.update(get_metrics(fake_s, real_s), fake_s.shape[0])
        
        print('Iter: {} PSNR: {:.4f}'.format(i, cur_psnr))
        
        if opt.save_image:
            save_image(fake_s, single_dir + '/' + str(i).zfill(4) + '.png')
            save_image(real_b, multiple_dir + '/' + str(i).zfill(4) + '_b.png')
            save_image(fake_s, multiple_dir + '/' + str(i).zfill(4) + '_r.png')
            save_image(real_s, multiple_dir + '/' + str(i).zfill(4) + '_s.png')
        
    print('Average PSNR: {:.4f}'.format(psnr_meter.average()))
    
if __name__ == '__main__':
    main()
