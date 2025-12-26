import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from src.dataset import dataset_loader
from src.model import VisionTransformer

def train():
    token_dim = 256
    num_classes = 10
    img_size = 32
    patch_size = 4
    num_patches = (img_size//patch_size)**2
    transformer_blocks = 8
    learning_rate = 5e-4

    epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load datasets
    dataset = dataset_loader()
    train_loader, test_loader = dataset.load_data(".\data", batchsize = 128)

    #model
    model=VisionTransformer(token_dim, num_classes, patch_size, num_patches, transformer_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = 0.05)
    criterion =  nn.CrossEntropyLoss()
    # 'steps_per_epoch' is the number of batches in your DataLoader
    # 'epochs' is the total number of training loops
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=5e-4, 
        steps_per_epoch=len(train_loader), 
        epochs=100,
        pct_start=0.1,    # Spend 10% of time warming up
        anneal_strategy='cos' # Cosine decay
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_epoch = 0
        total_epoch = 0
        print(f"\nEpoch {epoch+1}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # IMPORTANT: OneCycleLR steps after every BATCH, not every epoch
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            accuracy = 100.0 * correct / labels.size(0)

            correct_epoch += correct
            total_epoch += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx+1:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%")

        epoch_acc = 100.0 * correct_epoch / total_epoch
        print(f"==> Epoch {epoch+1} Summary: Total Loss = {total_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
    
    save_path = "./checkpoints/vit_cifar10_8b_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

if __name__ == '__main__':
    # This is the "proper idiom" the error is talking about
    train()