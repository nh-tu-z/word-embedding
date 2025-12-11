import os
import requests
import zipfile
import re
from collections import Counter

def download_ptb_dataset(url="https://d2l-data.s3-accelerate.amazonaws.com/ptb.zip", extract_path="."):
    zip_path = os.path.join(extract_path, "ptb.zip")
    if not os.path.exists(zip_path):
        print("Downloading PTB dataset...")
        r = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(r.content)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    print("PTB dataset ready.")

def preprocess_text(file_path, num_tokens=10000):
    with open(file_path, 'r') as f:
        text = f.read().lower()
    tokens = re.findall(r"[a-z]+", text)
    counter = Counter(tokens)
    vocab = [word for word, freq in counter.most_common(num_tokens)]
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    corpus = [word for word in tokens if word in word_to_idx]
    return corpus, vocab, word_to_idx

def generate_pairs(corpus, window_size=2):
    pairs = []
    for i, center in enumerate(corpus):
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
            if i != j:
                pairs.append((center, corpus[j]))
    return pairs

def plot_loss(losses, title, filename):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(range(len(losses)), losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'outputs/{filename}')
    print(f"Loss chart saved at outputs/{filename}")

def visualize_embeddings(embeddings, vocab, filename):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    for i, word in enumerate(vocab):
        plt.scatter(emb_2d[i,0], emb_2d[i,1])
        plt.annotate(word, (emb_2d[i,0], emb_2d[i,1]))
    plt.title('t-SNE Visualization of Embeddings')
    plt.savefig(f'outputs/{filename}')
    print(f"Embedding visualization saved at outputs/{filename}")
    
def generate_cbow_pairs(corpus, window_size=2):
    pairs = []
    for i in range(window_size, len(corpus) - window_size):
        context = corpus[i - window_size:i] + corpus[i + 1:i + window_size + 1]
        center = corpus[i]
        pairs.append((context, center))
    return pairs