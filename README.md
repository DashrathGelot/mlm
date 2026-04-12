# mlm

### Pretraining vs Finetuning
- Pretraining is the phase where LLM learns the structure and knowledge of language by predicting text on massive data.
- Finetuning is the process of adapting pretrained LLM to a specific task or domain by training it on trageted data.

### Instruction Finetuning vs Supervised Finetuning
- Intruction Finetuning: Teaches model to follow instructions and generate text. for example: summarize given text, translate text, answer questions, etc.
- Supervised Finetuning: fine tuning method that uses labeled data to train a model to perform a specific task, for example text classification, question -> sql

### Encoder:
- encoder is a part of neural network that processes input text into a fixed size vector representation.

### Decoder:
- Decoder is part of neural network that processes the vector representation into output text.

### Transformer:
- Transformer is a type of neural network architecture designed to handle sequences of data such as text by focusing on relationships between all words in the sequence using attention mechanism.
- A transformer reads a sentence all at once and decides "which words matter to each other" using attention mechanism.
- A transformer is neural network that uses self attention to understand relationships between all parts of input data simultaneously.

### Zero shot vs one shot vs few shot learning
- Zero shot: with zero short it predicts the output based on the natural language instruction without any example
- One Shot: with one short model predicts the output based on the natural language instruction and one example.
- Few shot: with few shot model predicts the output based on the natural language instruction and few examples.

### Context Size:
- How many tokens model can process or give input at once.

### Tensors:
- Tensor is multidimensional array with gpu support and automatic differentiation, used as the fundamentals building blocks of deeplearning models.