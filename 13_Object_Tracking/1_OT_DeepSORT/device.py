import torch

# device = "cuda" if torch.cuda.is_available() else "cpu" # for Google Colab
device = "mps" if torch.backends.mps.is_available() else "cpu" # for Apple Silicon

if __name__ == "__main__":
    print("device: ", device)