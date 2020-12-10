
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

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

epochs = 2000


# create the encoder class
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(1024, 256)
        self.encoder_output_layer = nn.Linear(256, 128)
        self.prelu = nn.PReLU(1, 0.25)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = self.prelu(activation)
        code = self.encoder_output_layer(activation)
        code = self.prelu(code)
        return code

# create the decoder class
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_hidden_layer = nn.Linear(128, 256)
        self.decoder_output_layer = nn.Linear(256, 1024)
        self.prelu = nn.PReLU(1, 0.25)

    def forward(self, features):
        activation = self.decoder_hidden_layer(features)
        activation = self.prelu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = self.prelu(activation)
        return reconstructed

# use the encoder and decoder class to build the autoencoder
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

# setup everything
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(device)

for parameter in model.decoder.parameters():
    parameter.requires_grad = False

#optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.MSELoss()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="D:/data", train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

#cv2.setNumThreads(0)

#torch.multiprocessing.set_start_method('spawn')
if __name__ == '__main__':

    rng = np.random.default_rng()
    x = rng.integers(-2048, 2048, size=(1, 1, 1, 1024), dtype=np.int16, endpoint=False).astype(np.float32)
    X = torch.from_numpy(x)
    X = X.view(-1, 1024).to(device)

    for epoch in range(epochs):
        loss = 0
        optimizer.zero_grad()
        outputs = model(X)
        #outputs = torch.floor(outputs + 0.5)

        train_loss = criterion(outputs, X)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


bp = 2

outputs = model(X)
outputs = torch.floor(outputs + 0.5)
loss = criterion(outputs, X)

print("loss = {:.6f}".format(loss.item()))

bp = 5
