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

tokenizer = SimpleTokenizerV1(build_vocab())
text = """
It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
"""
token_ids = tokenizer.encode(text)
print(token_ids)
print(tokenizer.decode(token_ids))
