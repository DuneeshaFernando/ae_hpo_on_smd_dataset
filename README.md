This repo is used to keep code that hyperparameter tunes the various training  approaches of the SMD dataset.
The primary ML approach for training the models is AE.

1) Training each machine separately - This is used to evaluate the different ML approaches as well
2) Training machines within a cluster using TL
3) Training a general model for all machines

Training each machine separately - Pick the middle one from each group (which makes it 3) and hyperparameter tune those 3.
Try to find common hyperparameters, and use those as the hyperparameters for models of all 28 machines.

