from tokenization import build_vocab
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        return [self.str_to_int[token] for token in preprocessed]
    
    def decode(self, token_ids):
        text = " ".join([self.int_to_str[token_id] for token_id in token_ids])
        # Replace spaces before punctuation
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

def run_tokenizer_v1():
    tokenizer = SimpleTokenizerV1(build_vocab())
    text = """
    It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
    """
    token_ids = tokenizer.encode(text)
    print(token_ids)
    print(tokenizer.decode(token_ids))

## Special context token Problem:
## here above tokenizer has one problem which is what if we have new word in the text which is not in the vocab?
## for example: "Hello, my name is Dashrath Gelot"
## It will throw an error because "Dashrath" and "Gelot" are not in the vocab.
## Solution: Add special tokens to the vocab.
## <UNK> token is used to represent unknown words.
## <|endoftext|> token is used to represent the end of the text.

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        return [self.str_to_int.get(token, self.str_to_int["<|unk|>"]) for token in preprocessed]
    
    def decode(self, token_ids):
        text = " ".join([self.int_to_str[token_id] for token_id in token_ids])
        # Replace spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'|])', r'\1', text)
        return text.replace("<|endoftext|>", "\n")

def run_tokenizer_v2():
    tokenizer = SimpleTokenizerV2(build_vocab())
    text = "Hello, Mrs. Gisburn drew back the window-curtains unknown"
    text2 = "It's the last he painted, you know,"
    text = text + " <|endoftext|> " + text2
    token_ids = tokenizer.encode(text)
    print(token_ids)
    print(tokenizer.decode(token_ids))

## Above is word based tokenization 
## Problem: It is very difficult to handle unknown words and it is very difficult to handle rare words.
## Solution: Byte Pair Encoding (BPE)
## BPE Algorithm: Learn frequent subwords automatically from data
## core idea: repetedly merge most frequent byte pair of symbols into a new symbol
"""
1. Start with corpus text
2. Split everything into characters/bytes
3. Add special end-of-word symbol (_)

4. Repeat until vocab size = target:
    a. Count frequency of all adjacent symbol pairs
    b. Find most frequent pair (A, B)
    c. Merge it → AB (new token)
    d. Replace all occurrences of (A, B) with AB
    e. Add AB to vocabulary
"""

# Let's Use tiktoken to utilize BPE
import tiktoken

def bpe_tokenizer(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text)
    return token_ids

def bpe_detokenizer(token_ids):
    tokenizer = tiktoken.get_encoding("gpt2")
    text = tokenizer.decode(token_ids)
    return text

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

def run_bpe_tokenizer():
    text = "Hello, Mrs. Gisburn drew back the window-curtains unknown"
    token_ids = bpe_tokenizer(text)
    print(token_ids)
    decoded_text = bpe_detokenizer(token_ids)
    print(decoded_text)