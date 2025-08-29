from torch.utils.data import Dataset



'''
diff path dataset is a dataset that contains diffusion paths 
for each sample, it contains the following:
- generated image VAE(X1)
- condition prompt
- diffusion path T0 -> T1 (sigmas), X0 -> X1 (latents)
- inference config
'''


