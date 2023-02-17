from torchvision import datasets, transforms
from pathlib import Path

def main():
    datasets_path = Path("datasets")

    train_dataset = datasets.DTD(root=datasets_path,
                                split="train",
                                transform=transforms.ToTensor(),
                                download=True)

    # val_dataset = datasets.DTD(root=datasets_path,
    #                         split="val",
    #                         transform=transforms.ToTensor(),
    #                         download=True)

    # test_dataset = datasets.DTD(root=datasets_path,
    #                             split="test",
    #                             transform=transforms.ToTensor(),
    #                             download=True)

if __name__ == "__main__":
    main()
