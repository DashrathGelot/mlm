# Creating Input Output target Pair for training Language Model

"""
Input: I -> Output:  H
Input: I H -> Output: AD
Input: I HAD -> Output:  always
Input: I HAD always -> Output:  thought
-----------------------------------------
Input: [40] -> Output: 367
Input: [40, 367] -> Output: 2885
Input: [40, 367, 2885] -> Output: 1464
Input: [40, 367, 2885, 1464] -> Output: 1807
"""

from tokenizer import get_tokenizer
from tokenization import read_data
class SimpleDataloader:
    def __init__(self, text, tokenizer, context_size=4):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    def demo(self):
        sample_tokens = self.tokens[:50]
        for i in range(self.context_size):
            input = sample_tokens[:i+1]
            output = sample_tokens[i+1]
            # print(f"Input: {input} -> Output: {output}")
            print(f"Input: {self.tokenizer.decode(input)} -> Output: {self.tokenizer.decode([output])}")

raw_data = read_data()
dl = SimpleDataloader(raw_data, get_tokenizer(), context_size=4)
# dl.demo()

## Tensor Dataset and Dataloader for generating or enriching training data by putting data in tensors
import torch
from torch.utils.data import DataLoader, Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(tokens) - context_size, stride):
            input_chunk = tokens[i:i+context_size]
            output_chunk = tokens[i+1: i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, id):
        return self.input_ids[id], self.target_ids[id]

def create_dataloader_v1(text, tokenizer, batch_size=4, context_size=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(text, tokenizer, context_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


def demo():
    ## batch size 1 and stride 1
    dataloader = create_dataloader_v1(raw_data, get_tokenizer(), batch_size=1, context_size=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    print("===============")

    ## batch size 4, and stride 4
    dataloader = create_dataloader_v1(raw_data, get_tokenizer(), batch_size=4, context_size=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    input, targets = next(data_iter)
    print("Inputs:\n", input)
    print("\nOutput:\n", targets)

# demo()