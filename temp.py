import torch

data = torch.load("train.pt")

data.split_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
data.split_embeddings = 1

data.abc = 1

pass