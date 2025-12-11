import argparse
import torch
from torch import nn
import torch.optim as optim
import time
from collections import Counter
import heapq
from utils import download_ptb_dataset, preprocess_text, generate_pairs, plot_loss
import os

os.makedirs('outputs', exist_ok=True)

# --- 1. Refine Node Structure to store fixed indices ---
class HuffmanNode:
    def __init__(self, freq, idx=None, left=None, right=None):
        self.freq = freq
        self.idx = idx # Word index (if leaf) or internal node index
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

# --- 2. Refine Tree Building to index internal nodes ---
def build_huffman_tree(word_freq, vocab_size):
    # Create leaf nodes for words in the vocabulary (indices 0 -> vocab_size - 1)
    heap = [HuffmanNode(freq, idx=i) for i, (word, freq) in enumerate(word_freq.items())]
    heapq.heapify(heap)
    
    # Start indexing internal nodes starting from vocab_size
    internal_node_idx = vocab_size 
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create a new parent node with the next available index
        merged = HuffmanNode(left.freq + right.freq, idx=internal_node_idx, left=left, right=right)
        internal_node_idx += 1 # Increment index for the next internal node
        
        heapq.heappush(heap, merged)
        
    return heap[0] # Return root

# Function to assign codes; stores path as a list of integer indices instead of objects
def assign_codes(node, code="", path=[], codes={}, paths={}):
    if node.left is None and node.right is None: # Is a leaf node
        codes[node.idx] = code
        paths[node.idx] = path # Store list of parent node indices
    else:
        # When traversing down, add current node index to the path
        new_path = path + [node.idx]
        assign_codes(node.left, code + "0", new_path, codes, paths)
        assign_codes(node.right, code + "1", new_path, codes, paths)
    return codes, paths

# --- Data Preparation ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_tokens", type=int, default=1000) # Limit tokens for quick demo
args, unknown = parser.parse_known_args() # Use parse_known_args for Colab/Notebook compatibility

download_ptb_dataset()
corpus, vocab, word_to_idx = preprocess_text("ptb/ptb.train.txt", num_tokens=args.num_tokens)
pairs = generate_pairs(corpus)

word_freq = Counter(corpus)
vocab_size = len(vocab)

# Build tree and assign standard indices
root = build_huffman_tree(word_freq, vocab_size)
codes, paths = assign_codes(root)

# --- 3. Refined Model (Added Epsilon and Standard Indexing) ---
class HierarchicalSoftmax(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.center_embeddings = nn.Embedding(vocab_size, embed_size)
        # The number of internal nodes in a Huffman tree with N leaves is N-1.
        # Total nodes requiring embedding is approx 2*N.
        self.node_embeddings = nn.Embedding(2 * vocab_size, embed_size)

    def forward(self, center_idx, path_indices, code_bits):
        # center_idx: [1]
        center_embed = self.center_embeddings(center_idx) # [1, embed_size]
        
        # Convert path_indices to tensor to fetch embeddings in one go (Batch processing)
        path_tensor = torch.tensor(path_indices, dtype=torch.long) # [path_len]
        node_embeds = self.node_embeddings(path_tensor) # [path_len, embed_size]
        
        # Calculate dot product between center and all nodes in the path
        # [1, embed_size] * [embed_size, path_len] -> [1, path_len]
        scores = torch.sigmoid(torch.matmul(center_embed, node_embeds.t())).squeeze()
        
        loss = 0
        epsilon = 1e-9 # CRITICAL FIX: Avoid log(0) resulting in NaN
        
        # Iterate through bits to calculate loss
        for i, bit in enumerate(code_bits):
            # Handle cases where path length might be 1 (scalar score)
            score = scores[i] if len(scores.shape) > 0 else scores 
            if bit == "1":
                loss += -torch.log(score + epsilon)
            else:
                loss += -torch.log(1 - score + epsilon)
                
        return loss

# --- Training Loop ---
embed_size = 50 # Demo embedding size
lr = 0.001
epochs = 5

model = HierarchicalSoftmax(len(vocab), embed_size)
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
start_time = time.time()

print("Start Training...")
for epoch in range(epochs):
    total_loss = 0
    # Run first 5000 pairs for testing; remove [:5000] for full dataset run
    for i, (center, context) in enumerate(pairs[:50000]): 
        center_idx = torch.tensor([word_to_idx[center]], dtype=torch.long)
        context_idx = word_to_idx[context]
        
        # Skip if context word is not in vocab (rare words that might have been pruned)
        if context_idx not in paths: continue 
        
        # Retrieve pre-calculated path and codes
        path_nodes = paths[context_idx] # List of integers
        code_bits = codes[context_idx]  # String "010..."
        
        optimizer.zero_grad()
        loss = model(center_idx, path_nodes, code_bits)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")

# Use plot function if available in utils
try:
    plot_loss(losses, 'Hierarchical Softmax Loss', 'loss_hierarchical_softmax.png')
    visualize_embeddings(model.center_embeddings.weight.detach().numpy(), vocab[:200], 'tsne_hsoftmax.png')
except:
    pass