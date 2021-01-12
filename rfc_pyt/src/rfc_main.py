
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import math
import numpy as np

'''
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )
'''

###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

max_epochs = 10000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
input_size = 8
feature_size = 2
decoder_int1 = 256

# create the encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 16, bias=False)
        self.hidden_layer_1 = nn.Linear(16, 16, bias=False)
        self.hidden_layer_2 = nn.Linear(128, 128, bias=False)
        #self.hidden_layer_3 = nn.Linear(32, 128, bias=False)
        self.output_layer = nn.Linear(16, feature_size, bias=False)
        self.prelu = nn.PReLU(1, 0.25)
        self.silu = nn.SiLU()
        self.elu = nn.ELU()
        self.tanshrink = nn.Tanhshrink()

    def forward(self, activation):
        activation = self.input_layer(activation)
        activation = self.prelu(activation)
        #activation = self.hidden_layer_1(activation)
        #activation = self.silu(activation)
        #activation = self.elu(activation)
        #activation = self.prelu(activation)
        #activation = self.tanshrink(activation)
        #activation = self.hidden_layer_2(activation)
        #activation = self.prelu(activation)
        #activation = self.hidden_layer_3(activation)
        #activation = self.prelu(activation)
        activation = self.output_layer(activation)
        #activation = self.prelu(activation)
        #activation = self.silu(activation)
        #activation = self.elu(activation)
        #activation = self.prelu(activation)
        #activation = self.tanshrink(activation)
        return activation

# create the decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(feature_size, decoder_int1, bias=False)
        self.hidden_layer_1 = nn.Linear(8, 8, bias=False)
        self.hidden_layer_2 = nn.Linear(16, 16, bias=False)
        self.output_layer = nn.Linear(decoder_int1, output_size, bias=False)
        self.prelu = nn.PReLU(1, 0.25)
        self.multp = nn.Parameter(torch.tensor([[2048.0]]))
        self.silu = nn.SiLU()
        self.elu = nn.ELU()
        self.tanshrink = nn.Tanhshrink()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, activation):
        #activation = self.prelu(activation)
        activation = self.input_layer(activation)
        #activation = self.elu(activation)
        #activation = self.prelu(activation)
        #activation = self.relu(activation)
        #activation = activation*self.multp.expand_as(activation)
        #activation = self.hidden_layer_1(activation)

        #activation = self.hidden_layer_2(activation)
        #activation = self.silu(activation)

        activation = self.output_layer(activation)
        #activation = self.tanh(activation)
        #activation = self.prelu(activation)
        #activation = self.silu(activation)
        #activation = self.elu(activation)
        #activation = activation
        return activation

# use the encoder and decoder classes to build the autoencoder
class AE(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.encoder = Encoder(input_size, feature_size)
        self.decoder = Decoder(input_size, feature_size)

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

# setup everything
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE(input_size, feature_size).to(device)

# cycle through the decoder side and do whatever needs doing
for parameter in model.decoder.parameters():
    parameter.requires_grad = False
    p = parameter.data
    bp = 0

model.decoder.prelu.weight.requires_grad = False

# use something like this to manually set the weights.  use the no_grad() to prevent tracking of gradient changes
with torch.no_grad():
    rnd_range = 0.6
    mr = np.random.default_rng(1)
    model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(mr.uniform(-rnd_range, rnd_range, [decoder_int1, feature_size]).astype(np.float32))).to(device)
    model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(mr.uniform(-rnd_range, rnd_range, [input_size, decoder_int1]).astype(np.float32))).to(device)
    #model.decoder.output_layer.weight.data = nn.Parameter(torch.tensor((0.5 + 0.5)*torch.rand((input_size, 128)) - 0.5)).to(device)
    #model.decoder.output_layer.weight.data = nn.Parameter(torch.tensor([[1.0166], [1.3116], [1]])).to(device)
    #model.decoder.output_layer.bias.data = nn.Parameter(torch.tensor((2 + 2) * torch.rand((input_size)) - 2)).to(device)
    #model.decoder.output_layer.weight.data = nn.Parameter(0.25*torch.ones(input_size, 128)).to(device)
    #model.decoder.output_layer.bias.data = nn.Parameter(0.1*torch.ones(input_size)).to(device)
    #model.decoder.prelu.weight.data = nn.Parameter(torch.tensor(0.05*torch.ones(1))).to(device)

    #model.encoder.input_layer.weight.data = nn.Parameter(torch.tensor([0.37, 0.47])).to(device)
    ew1 = copy.deepcopy(model.encoder.input_layer.weight)
    dw1 = copy.deepcopy(model.decoder.output_layer.weight)


# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, div_factor=100, steps_per_epoch=1, epochs=max_epochs, verbose=True)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5000, threshold=0.1, threshold_mode='rel', cooldown=20, min_lr=1e-10, eps=1e-08, verbose=True)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()

'''
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="D:/data", train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
'''

#cv2.setNumThreads(0)

#torch.multiprocessing.set_start_method('spawn')
if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng()
    x = rng.integers(-8, 8, size=(1, 1, 1, input_size), dtype=np.int16, endpoint=False).astype(np.float32)
    #x = np.fromfile("../../../rf_zsl/data/brf_test_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
    #x = x[np.s_[idx:idx+input_size]]
    X = torch.from_numpy(x).to(device)
    X = X.view(-1, input_size)

    for epoch in range(max_epochs):
        loss = 0
        optimizer.zero_grad()
        outputs = model(X)
        #outputs = torch.floor(outputs + 0.5)

        train_loss = criterion(outputs, X)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, (loss)))

        if (torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1):
            break

        #scheduler.step(math.floor(loss))
        #scheduler.step()

    with torch.no_grad():
        #outputs = model(X)
        #outputs = torch.floor(outputs + 0.5)
        loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        print("\nloss = {:.6f}".format(loss.item()))

        ew2 = copy.deepcopy(model.encoder.input_layer.weight)
        dw2 = copy.deepcopy(model.decoder.output_layer.weight)

        bp = 5
        #print("\nOriginal Input:\n", X)
        #print("\nOutput:\n",torch.floor(outputs + 0.5))

        f = model.encoder(X)

        D = model.decoder
        Y = torch.floor(D(f) + 0.5)

    bp = 9
