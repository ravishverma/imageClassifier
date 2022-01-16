# Semi-supervised Image Classification
![model](./projectPlan/model.png)

## Workflow
  - Train the autoencoder: `python trainAutoencoder.py`
  - Create low dimensional representation: `python predictAutoencoder.py <checkpoint file>`
  - Create clusters: `python classify.py`<algorithm>`
  - Check the images in each cluster: `python viewResults.py`

## Current State
### Reconstruction Loss on Autoencoder
![model](./results/trainAutoencoder.png)

### Latent Space Distribution
![model](./results/assessLatentSpace.png)

### Latent Space Clusters
![model](./results/clustersLatentSpace.png)
