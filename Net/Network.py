from Net.Pseudo_Siamese import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Siamese_1 = Siamese_Branch_A

        self.Siamese_2 = Siamese_Branch_B

        self.Gen_Module = Generation_Model

    def forward(self, img1, img2):
        F1 = self.Siamese_1(img1)
        F2 = self.Siamese_2(img2)

        F = torch.cat((F1, F2), 1)
        result = self.Gen_Module(F)
        return result