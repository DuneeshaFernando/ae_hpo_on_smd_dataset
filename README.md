This repo is used to keep code that hyperparameter tunes the AE-based training approach of the SMD dataset.

1) Training each machine separately - This is used to evaluate the different ML approaches as well
2) Training machines within a cluster
3) Training a general model for all machines

Time efficient approach for training each machine separately is to pick the middle one from each group (which makes it 3) and hyperparameter tune those 3.
Then try to find common hyperparameters, and use those as the hyperparameters for models of all 28 machines. However since the SMD dataset had concept drift 
issues, we had to perform hyperparameter tuning for all machines separately.

