## Repository of the paper "Examining the Effect of News Context on Algorithmic Trading"

#### Environment Details
1. python 3.8.12
2. transformers 4.31.0
3. torch 2.1.2
4. gym 0.21.0

#### Directory Information
lm directory: consists of the language models BERT and FinBERT

llm directory: consists of the large language models Llama2 and Mistral

#### Embedding Information
In case of BERT and FinBERT the trading environment generates the embeddings on the fly using the HuggingFace models.

In case of Llama 2 and Mistral, we generate the embeddings separately and store it in a .json file. The .json file contains the news title and embedding.

We use the .json file in the trading environment of llm directory.

#### Environment details for generating the LLM embeddings
1. python 3.11
2. transformers 4.39.0
3. autoawq 0.2.0

#### Script Details

cd <folder_name>

python train_agent.py --text_representation <text_embedding_model>
