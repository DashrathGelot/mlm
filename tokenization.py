"""
Tokenization is the process of splitting text into smaller units called tokens.
tokens converted into token_ids. (integers or floating point numbers)
Steps:
- Take data
- Split data into tokens
- Preprocess data (removing space....)
- Create vocabulary
- Convert tokens into token_ids
"""
import re

def read_data():
    with open("the-verdict.txt", "r") as f:
        raw_text = f.read()
    return raw_text

def split_data(raw_text):
    return re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)


def build_vocab():
    raw_text = read_data()
    tokens = split_data(raw_text)
    ## Remove empty tokens
    ## Important: It's not always obvious to remove space think about python code space matters
    ## to simplicity we are removing spaces
    result = [token for token in tokens if token.strip()]
    all_words = sorted(set(result))
    vocab = {token:i for i, token in enumerate(all_words)}
    return vocab

# raw_text = read_data()
# print(f"Total characters: {len(raw_text)}")
# print(raw_text[:99])

# tokens = split_data(raw_text)
# print(f"Total tokens: {len(tokens)}")
# print(tokens[:99])

# ## Remove empty tokens
# ## Important: It's not always obvious to remove space think about python code space matters
# ## to simplicity we are removing spaces
# result = [token for token in tokens if token.strip()]
# print(result[:99])

# ## Prepare vocabulary
# all_words = sorted(set(result))
# vocab_size = len(all_words)

# print(vocab_size)
# vocab = {token:i for i, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break