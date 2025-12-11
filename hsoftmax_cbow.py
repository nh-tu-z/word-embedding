import argparse
import torch
from torch import nn
import torch.optim as optim
import time
from collections import Counter
import heapq
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Assuming these functions are in your utils.py
from utils import download_ptb_dataset, preprocess_text, generate_cbow_pairs, plot_loss

os.makedirs('outputs', exist_ok=True)

# --- 1. Visualization Function (t-SNE) ---
def visualize_embeddings(embeddings, vocab, filename):
    # Limit the number of words to plot (Top 150) for clarity
    limit = min(150, len(vocab))
    embeddings = embeddings[:limit]
    vocab = vocab[:limit]
    
    # Use t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(vocab):
        plt.scatter(emb_2d[i, 0], emb_2d[i, 1])
        plt.annotate(word, (emb_2d[i, 0], emb_2d[i, 1]))
        
    plt.title('t-SNE Visualization of CBOW Embeddings')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'outputs/{filename}')
    print(f"Embedding visualization saved at outputs/{filename}")
    plt.show()

# --- 2. Huffman Tree Structure (Fixed Indexing) ---
class HuffmanNode:
    def __init__(self, freq, idx=None, left=None, right=None):
        self.freq = freq
        self.idx = idx  # Store fixed index (int) instead of object
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(word_freq, vocab_size):
    # Create Leaf nodes for vocabulary words (index 0 -> vocab_size-1)
    heap = [HuffmanNode(freq, idx=i) for i, (word, freq) in enumerate(word_freq.items())]
    heapq.heapify(heap)
    
    # Internal nodes start indexing from vocab_size onwards
    internal_node_idx = vocab_size 
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create parent node
        merged = HuffmanNode(left.freq + right.freq, idx=internal_node_idx, left=left, right=right)
        internal_node_idx += 1
        heapq.heappush(heap, merged)
        
    return heap[0]

def assign_codes(node, code="", path=[], codes={}, paths={}):
    if node.left is None and node.right is None: # Is leaf node
        codes[node.idx] = code
        paths[node.idx] = path # Store list of indices (int)
    else:
        new_path = path + [node.idx]
        assign_codes(node.left, code + "0", new_path, codes, paths)
        assign_codes(node.right, code + "1", new_path, codes, paths)
    return codes, paths

# --- 3. Data Preparation ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_tokens", type=int, default=1000) 
args, unknown = parser.parse_known_args()

print("Downloading and preprocessing data...")
download_ptb_dataset()
corpus, vocab, word_to_idx = preprocess_text("ptb/ptb.train.txt", num_tokens=args.num_tokens)
pairs = generate_cbow_pairs(corpus) # Use pair generation function for CBOW

word_freq = Counter(corpus)
vocab_size = len(vocab)

print("Building Huffman Tree...")
root = build_huffman_tree(word_freq, vocab_size)
codes, paths = assign_codes(root)

# --- 4. CBOW Hierarchical Softmax Model (NaN Fixed) ---
class CBOWHierarchicalSoftmax(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Embedding for context words
        self.context_embeddings = nn.Embedding(vocab_size, embed_size)
        # Embedding for Huffman tree nodes
        self.node_embeddings = nn.Embedding(2 * vocab_size, embed_size)

    def forward(self, context_idx, path_indices, code_bits):
        # 1. Calculate hidden vector h by averaging context vectors
        # context_idx: [batch_size, context_window * 2] -> here batch=1 so it is [len_context]
        embeds = self.context_embeddings(context_idx) # [len_context, embed_size]
        h = embeds.mean(dim=0).unsqueeze(0) # [1, embed_size]
        
        # 2. Get embeddings of nodes along the path
        path_tensor = torch.tensor(path_indices, dtype=torch.long)
        node_embeds = self.node_embeddings(path_tensor) # [path_len, embed_size]
        
        # 3. Calculate Dot product
        # [1, embed_size] * [embed_size, path_len] -> [1, path_len]
        scores = torch.sigmoid(torch.matmul(h, node_embeds.t())).squeeze()
        
        loss = 0
        epsilon = 1e-9 # CRITICAL: Prevent log(0) -> NaN
        
        # 4. Calculate Loss based on Huffman codes
        for i, bit in enumerate(code_bits):
            # Handle case where path length is 1 (scalar score)
            score = scores[i] if len(scores.shape) > 0 else scores
            
            if bit == "1":
                loss += -torch.log(score + epsilon)
            else:
                loss += -torch.log(1 - score + epsilon)
                
        return loss

# --- 5. Training Loop ---
embed_size = 50
lr = 0.001 # Reduce LR to avoid loss instability
epochs = 5

model = CBOWHierarchicalSoftmax(len(vocab), embed_size)
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
start_time = time.time()

print(f"Start training CBOW H-Softmax with vocab size: {len(vocab)}...")

for epoch in range(epochs):
    total_loss = 0
    # Run demo 5000 pairs. Remove [:5000] to run full dataset
    for i, (context, center) in enumerate(pairs[:5000]):
        # Convert context (list of words) to tensor indices
        context_idx = torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
        center_idx = word_to_idx[center]
        
        # Check if center word is in Huffman tree (handling rare words)
        if center_idx not in paths: continue
        
        path_nodes = paths[center_idx]
        code_bits = codes[center_idx]
        
        optimizer.zero_grad()
        loss = model(context_idx, path_nodes, code_bits)
        loss.backward()
        
        # Gradient Clipping: Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")

# --- 6. Plotting and Visualization ---
# 1. Plot Loss chart
try:
    plot_loss(losses, 'CBOW Hierarchical Softmax Loss', 'outputs/loss_cbow_hsoftmax.png')
except Exception as e:
    print(f"Could not plot loss: {e}")

# 2. Visualize Embeddings using t-SNE
print("Preparing visualization...")

# Get weights from context_embeddings layer and convert to numpy
embeddings_matrix = model.context_embeddings.weight.detach().cpu().numpy()

# Create vocabulary list in correct index order
idx_to_word = {i: w for w, i in word_to_idx.items()}
full_vocab_list = [idx_to_word[i] for i in range(len(idx_to_word))]

# Call visualization function
visualize_embeddings(embeddings_matrix, full_vocab_list, 'tsne_cbow_hsoftmax.png')