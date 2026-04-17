from tokenization import read_data
from dataloader import create_dataloader
import torch

vocab_size = 50257
dimension = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, dimension)

# print("Token Embedding Layer:", token_embedding_layer(torch.tensor([1])))
print("Token Embedding Layer Shape:", token_embedding_layer.weight.shape)

context_length = 4
raw_data = read_data()
dataloader = create_dataloader(raw_data, batch_size=8, context_size=context_length, stride=context_length, shuffle=False)

data_iter = iter(dataloader)
input, target = next(data_iter)

print("Token IDs:\n", input)
print("\nTarget IDs:\n", target)

print("\nInput Shape:", input.shape)

token_embeddings = token_embedding_layer(input)
print("\nToken Embeddings Shape:", token_embeddings.shape)


### Creating Positional Embeddings Vectors

positional_embedding_layer = torch.nn.Embedding(context_length, dimension)

positional_embeddings = positional_embedding_layer(torch.arange(context_length))
print("Positional Embeddings Shape:", positional_embeddings.shape)
print("Positional Embeddings:", positional_embeddings)
