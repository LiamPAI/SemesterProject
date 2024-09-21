import torch

def main():
    # PyTorch version and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Simple PyTorch operation
    x = torch.rand(5, 3)
    print("Random tensor:")
    print(x)

    # Test GPU if available
    if torch.cuda.is_available():
        print("Testing GPU:")
        device = torch.device("cuda")
        y = torch.rand(5, 3).to(device)
        print(y)

if __name__ == "__main__":
    main()
