import torch
from tqdm import tqdm

SOURCE_PATH = "data/raw/corruptmnist"
DEST_PATH = "data/processed/corruptmnist"


def normalize_data(data):
    normalized_tensors = []

    for img in tqdm(data):
        mean = img.mean()
        std_dev = img.std()

        # Avoid division by zero
        if std_dev == 0:
            normalized_tensor = img - mean
        else:
            normalized_tensor = (img - mean) / std_dev

        normalized_tensors.append(normalized_tensor)

    return normalized_tensors


if __name__ == "__main__":
    train = torch.tensor([])
    train_labels = torch.tensor([], dtype=torch.long)

    for i in range(6):
        train = torch.cat((train, torch.load(f"{SOURCE_PATH}/train_images_{i}.pt")))
        train_labels = torch.cat((train_labels, torch.load(f"{SOURCE_PATH}/train_target_{i}.pt")))

    data = normalize_data(train)
    labels = train_labels.flatten()

    torch.save({"data": data, "labels": labels}, f"{DEST_PATH}/train_set.pth")

    test = torch.load(f"{SOURCE_PATH}/test_images.pt")
    test_labels = torch.load(f"{SOURCE_PATH}/test_target.pt").long()
    data = normalize_data(test)
    labels = test_labels.flatten()

    torch.save({"data": data, "labels": labels}, f"{DEST_PATH}/test_set.pth")
