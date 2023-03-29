import pandas as pd
from sklearn import preprocessing
import src.constants as const
from os.path import join
import numpy as np
import torch
import torch.utils.data as data_utils
from config import config as conf
import src.autoencoder as autoencoder
from src.evaluation import Evaluation
from datetime import datetime

file_name="machine-1-2"
mod_name = "m12"
target_configs = {'NUM_LAYERS': 9, 'WINDOW_SIZE': 95, 'HIDDEN_SIZE': 14, 'BATCH_SIZE': 926, 'LEARNING_RATE': 0.000812013996755207}

# Pre-requisites
min_max_scaler = preprocessing.MinMaxScaler()

# setting seed for reproducibility
torch.manual_seed(conf.SEED)
np.random.seed(conf.SEED)

dataset_path = const.SMD_DATASET_LOCATION

# Read normal data
normal_path = join(dataset_path,'train/')
normal_data_file = join(normal_path, file_name+".csv")
normal_df = pd.read_csv(normal_data_file)
normal_df = normal_df.astype(float)

# Read anomaly data
anomaly_path = join(dataset_path,'test_with_labels/')
anomaly_data_file = join(anomaly_path, file_name+".csv")
anomaly_df = pd.read_csv(anomaly_data_file)
# Separate out the anomaly labels before normalisation/standardization
anomaly_df_labels = anomaly_df['Normal/Attack']
anomaly_df = anomaly_df.drop(['Normal/Attack'], axis=1)
anomaly_df = anomaly_df.astype(float)

# Normalise/ standardize the normal and anomaly dataframe
full_df = pd.concat([normal_df, anomaly_df])
min_max_scaler.fit(full_df)

normal_df_values = normal_df.values
normal_df_values_scaled = min_max_scaler.transform(normal_df_values)
normal_df_scaled = pd.DataFrame(normal_df_values_scaled)

anomaly_df_values = anomaly_df.values
anomaly_df_values_scaled = min_max_scaler.transform(anomaly_df_values)
anomaly_df_scaled = pd.DataFrame(anomaly_df_values_scaled)

# Preparing the datasets for training and testing using AutoEncoder
windows_normal = normal_df_scaled.values[np.arange(target_configs["WINDOW_SIZE"])[None, :] + np.arange(normal_df_scaled.shape[0] - target_configs["WINDOW_SIZE"])[:, None]]
windows_anomaly = anomaly_df_scaled.values[np.arange(target_configs["WINDOW_SIZE"])[None, :] + np.arange(anomaly_df_scaled.shape[0] - target_configs["WINDOW_SIZE"])[:, None]]

w_size = windows_normal.shape[1] * windows_normal.shape[2] # w_size is the input window size
z_size = target_configs["HIDDEN_SIZE"] # z_size is the latent size

windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 * windows_normal.shape[0])):]

# Create batches of training and testing data
train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], w_size]))
), batch_size=target_configs["BATCH_SIZE"], shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], w_size]))
), batch_size=target_configs["BATCH_SIZE"], shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_anomaly).float().view(([windows_anomaly.shape[0], w_size]))
), batch_size=target_configs["BATCH_SIZE"], shuffle=False, num_workers=0)

# Initialise the AutoEncoder model
autoencoder_model = autoencoder.AutoEncoder(in_size=w_size, latent_size=z_size, num_layers=target_configs["NUM_LAYERS"])
# Start training and save the best model, i.e. the model with the least validation loss
model_path = const.MODEL_LOCATION
model_name = join(model_path, "ae_model_"+mod_name+"_from_scratch.pth")
start_time = datetime.now()
autoencoder.training(conf.N_EPOCHS, autoencoder_model, train_loader, val_loader, target_configs["LEARNING_RATE"], model_name)
end_time = datetime.now()
print("Training time :", end_time-start_time)

# Load the model
checkpoint = torch.load(model_name)
autoencoder_model.encoder.load_state_dict(checkpoint['encoder'])
autoencoder_model.decoder.load_state_dict(checkpoint['decoder'])

# Use the trained model to obtain predictions for the test set
results = autoencoder.testing(autoencoder_model, test_loader)
y_pred_for_test_set = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(), results[-1].flatten().detach().cpu().numpy()])

# Process the actual labels
windows_labels = []
for i in range(len(anomaly_df_labels) - target_configs["WINDOW_SIZE"]):
    windows_labels.append(list(np.int_(anomaly_df_labels[i:i + target_configs["WINDOW_SIZE"]])))

processed_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

thresholding_percentile = 100 - (((processed_test_labels.count(1.0)) / (len(processed_test_labels))) * 100)

# Obtain threshold based on pth percentile of the mean squared error
threshold = np.percentile(y_pred_for_test_set, [thresholding_percentile])[0]  # 90th percentile

# Map the predictions to anomaly labels after applying the threshold
predicted_labels = []
for val in y_pred_for_test_set:
    if val > threshold:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

# Evaluate the predicted_labels against the actual labels
test_eval = Evaluation(processed_test_labels, predicted_labels)
test_eval.print()
