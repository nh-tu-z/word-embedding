import argparse
import torch
from torch import nn
import torch.optim as optim
import time
from utils import download_ptb_dataset, preprocess_text, generate_pairs, plot_loss, visualize_embeddings
import os

os.makedirs('outputs', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_tokens", type=int, default=10000)
args = parser.parse_args()

# Download and preprocess dataset
download_ptb_dataset()
corpus, vocab, word_to_idx = preprocess_text("ptb/ptb.train.txt", num_tokens=args.num_tokens)
pairs = generate_pairs(corpus)

# Model
class SkipGramSoftmax(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_size)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, center_idx):
        embed = self.center_embeddings(center_idx)
        return self.output_layer(embed)

embed_size = 100
lr = 0.01
epochs = 5  # PTB is large; keep epochs small for demo

model = SkipGramSoftmax(len(vocab), embed_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
start_time = time.time()
for epoch in range(epochs):
    total_loss = 0
    for center, context in pairs[:50000]:  # limit for demo
        center_idx = torch.tensor([word_to_idx[center]], dtype=torch.long)
        context_idx = torch.tensor([word_to_idx[context]], dtype=torch.long)
        optimizer.zero_grad()
        logits = model(center_idx)
        loss = criterion(logits, context_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
end_time = time.time()

plot_loss(losses, 'Skip-Gram Softmax Loss', 'loss_softmax.png')
visualize_embeddings(model.center_embeddings.weight.detach().numpy(), vocab[:200], 'tsne_softmax.png')
print(f"Training time: {end_time - start_time:.4f} seconds")