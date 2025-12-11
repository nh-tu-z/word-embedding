import argparse
import torch
from torch import nn
import torch.optim as optim
import time
from collections import Counter
from utils import download_ptb_dataset, preprocess_text, generate_cbow_pairs, plot_loss, visualize_embeddings
import os

os.makedirs('outputs', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_tokens", type=int, default=10000)
args = parser.parse_args()

download_ptb_dataset()
corpus, vocab, word_to_idx = preprocess_text("ptb/ptb.train.txt", num_tokens=args.num_tokens)
pairs = generate_cbow_pairs(corpus)

word_freq = Counter(corpus)
freqs = torch.tensor([word_freq[w] for w in vocab], dtype=torch.float)
neg_sampling_dist = freqs.pow(0.75)
neg_sampling_dist /= neg_sampling_dist.sum()

class CBOWNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.context_embeddings = nn.Embedding(vocab_size, embed_size)
        self.center_embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, context_idx, pos_idx, neg_idx):
        context_embed = self.context_embeddings(context_idx).mean(dim=0).unsqueeze(0)
        pos_embed = self.center_embeddings(pos_idx)
        neg_embed = self.center_embeddings(neg_idx)
        pos_score = torch.sigmoid(torch.matmul(context_embed, pos_embed.t()))
        neg_score = torch.sigmoid(torch.matmul(context_embed, neg_embed.t()) * -1)
        return pos_score, neg_score

embed_size = 100
lr = 0.01
epochs = 5
num_negatives = 5

model = CBOWNegSampling(len(vocab), embed_size)
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
start_time = time.time()
for epoch in range(epochs):
    total_loss = 0
    for context, center in pairs[:50000]:
        context_idx = torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
        pos_idx = torch.tensor([word_to_idx[center]], dtype=torch.long)
        neg_idx = torch.multinomial(neg_sampling_dist, num_negatives, replacement=True)
        optimizer.zero_grad()
        pos_score, neg_score = model(context_idx, pos_idx, neg_idx)
        loss = -torch.log(pos_score) - torch.sum(torch.log(neg_score))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
end_time = time.time()

plot_loss(losses, 'CBOW Negative Sampling Loss', 'loss_cbow_negative_sampling.png')
visualize_embeddings(model.context_embeddings.weight.detach().numpy(), vocab[:200], 'tsne_cbow_negative_sampling.png')
print(f"Training time: {end_time - start_time:.4f} seconds")