import torch
import torch.nn as nn
import src.trainer_utils as utils
from collections import OrderedDict

device = utils.get_default_device()

class AutoEncoder(nn.Module):
    def __init__(self, in_size, latent_size, num_layers):
        super().__init__()

        num_neurons=[]
        for l in range(num_layers):
            num_neurons.append(in_size)
            in_size=int(in_size/2)
        num_neurons.append(latent_size)

        encoder_layers = OrderedDict()
        for layer_n in range(num_layers):
            h_layer = nn.Linear(in_features=num_neurons[layer_n], out_features=num_neurons[layer_n+1])
            encoder_layers['layer_'+str(layer_n)] = h_layer
            encoder_layers['relu_'+str(layer_n)] = nn.ReLU()

        decoder_layers = OrderedDict()
        for layer_n in range(num_layers,0,-1):
            h_layer = nn.Linear(in_features=num_neurons[layer_n], out_features=num_neurons[layer_n-1])
            decoder_layers['layer_' + str(layer_n)] = h_layer
            if layer_n == 1:
                decoder_layers['sigmoid'] = nn.Sigmoid()
            else:
                decoder_layers['relu_' + str(layer_n)] = nn.ReLU()

        self.encoder = nn.Sequential(encoder_layers).to(device)
        self.decoder = nn.Sequential(decoder_layers).to(device)

    def forward(self, input_window):
        latent_window = self.encoder(input_window)
        reconstructed_window = self.decoder(latent_window)
        return reconstructed_window

def training(epochs, autoencoder_model, train_loader, val_loader, learning_rate, model_name):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    for epoch in range(epochs):
        train_loss = 0
        for [batch] in train_loader:
            batch = utils.to_device(batch, device)
            recon = autoencoder_model(batch)
            loss = criterion(recon, batch)
            train_loss += loss.item() * batch.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader.dataset)

        with torch.no_grad():
            val_loss = 0
            for [val_batch] in val_loader:
                val_batch = utils.to_device(val_batch, device)
                val_recon = autoencoder_model(val_batch)
                v_loss = criterion(val_recon, val_batch)
                val_loss += v_loss.item() * val_batch.shape[0]
            val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch:{epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving best model ..")
            # Save the model
            torch.save({
                'encoder': autoencoder_model.encoder.state_dict(),
                'decoder': autoencoder_model.decoder.state_dict()
            }, model_name)

def testing(autoencoder_model, test_loader):
    results = []
    for [batch] in test_loader:
        batch = utils.to_device(batch, device)
        with torch.no_grad():
            recon = autoencoder_model(batch)
        results.append(torch.mean((batch - recon) ** 2, axis=1))
    return results