import deeplake

#establish connection and store data
ds = deeplake.empty('hub://<ORGANIZATION_NAME>/gene_expression_dataset')

gene_expression_data = ds.create_tensor('gene_expression', htype='array', dtype='float32') 

import numpy as np

#access data
gene_expression_arrays = ...  # your gene expression data

with ds:
    for array in gene_expression_arrays:
        ds.gene_expression.append(array)


#Training a Model, replace with actual model

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Define your model
class GeneExpressionVAE(nn.Module):
    ...

vae = GeneExpressionVAE()

# Define loss function and optimizer
criterion = nn.MSELoss()  # or another appropriate loss function
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Load data into a PyTorch DataLoader
deeplake_loader = ds.pytorch(num_workers=0, batch_size=32, transform={'gene_expression': None}, shuffle=True)

# Training loop
for epoch in range(100):  # for some number of epochs
    for i, data in enumerate(deeplake_loader):
        gene_expression = data['gene_expression']
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        reconstructed, _, _ = vae(gene_expression)
        loss = criterion(reconstructed, gene_expression)
        loss.backward()
        optimizer.step()
