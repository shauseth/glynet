# GlyNet

GlyNet50 is deep neural network with 50 outputs for 50 different proteins.  
It takes encoded IUPAC strings of glycans as inputs.  
The library comes with high-level functions which can:
* Prepare the data with transformations and cutoffs.
* Perform cross-validation on the data with the given hyperparameters.
* Make a bar chart showing the r-squared for all the proteins.
* Make a scatter plot for each protein showing predictions vs ground-truth.
