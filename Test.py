import torch
import torch.optim
from Dataloader import Test_data_loader
from Net import Network
import torchvision
import time

def test(config):
    Net = Network.Net().cuda()
    Net.load_state_dict(torch.load(config.model_save_path + 'RACP_model.pth'))
    test_dataset = Test_data_loader(config.img1_test_path, config.img2_test_path, \
                               mode = 'test', img_height = 520, img_width = 520)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    Mean_time = []

    with torch.no_grad():
        for iteration, (i1, i2) in enumerate(test_loader):

            i1 = i1.cuda()
            i2 = i2.cuda()
            time_start = time.time()
            result_img = Net(i1, i2)
            time_end = time.time()
            print('time cost', time_end - time_start, 's')
            Mean_time.append(time_end - time_start)
            torchvision.utils.save_image(result_img, config.img_save_path + str(iteration) + '.png')

    print('Mean time cost:', sum(Mean_time)/len(Mean_time))
    del(Mean_time[0])
    print('Mean time cost (With out model loading):', sum(Mean_time)/len(Mean_time))