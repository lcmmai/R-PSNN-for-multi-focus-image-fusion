import torch
import torch.optim
from Dataloader import Train_data_loader
import Net.Network as Network
import torch.nn as nn
import numpy as np


def train(config):
    Net = Network.Net().cuda()
    train_dataset = Train_data_loader(config.img1_path, config.img2_path, config.groundtruth_path, \
                                      mode='train', img_height=350, img_width=350)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False,
                                               num_workers=1)
    criterion = nn.MSELoss().cuda()  # use MSE as loss function
    lr = config.lr
    optimizer = torch.optim.Adam(Net.parameters(), lr=lr, betas=(config.beta1, config.beta2))

    losstxt = "loss.txt"  # txt which record loss value

    Net.train()
    for epoch in range(1, config.sum_epochs):
        loss_sum = []

        if epoch % config.decay_epochs == 0:  # halve the lr each 30 epochs
            lr = lr / 2
            optimizer.param_groups[0]['lr'] = lr
        print('epoch number:', epoch, '-' * 20)
        for iteration, (i1, i2, gr) in enumerate(train_loader):
            i1 = i1.cuda()
            i2 = i2.cuda()
            gr = gr.cuda()

            optimizer.zero_grad()  # clear the gradient

            result_img = Net(i1, i2)
            loss = criterion(result_img, gr)  # calculate MSE loss
            loss.backward()  # back propagation, calculate the gradient
            optimizer.step()  # update parameters

            if iteration % 1 == 0:
                print("Loss at iteration", iteration, ":", loss.item())
                loss_sum.append(loss.item())  # add loss value behind list

        if epoch % config.save_epochs == 0:
            torch.save(Net.state_dict(), config.model_save_path + 'model_in_epoch' + str(epoch) + '.pth')
        loss_sum = np.array(loss_sum)
        with open(losstxt, 'a') as file:  # record average loss value
            file.write(str(np.mean(loss_sum)) + '\n')
