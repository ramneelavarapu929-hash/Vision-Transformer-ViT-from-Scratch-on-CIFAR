import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.model import VisionTransformer
from PIL import Image
def predict(img_path, model_path):
    token_dim = 256
    num_classes = 10
    img_size = 32
    patch_size = 4
    num_patches = (img_size//patch_size)**2
    transformer_blocks = 4
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = VisionTransformer(token_dim, num_classes, patch_size, num_patches, transformer_blocks)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        print(output)
        prediction = torch.argmax(output, dim=1)
    
    return prediction.item()
# CIFAR-10 Class Mapping
CIFAR10_CLASSES = {
    0: "airplane",1: "automobile",2: "bird",
    3: "cat",4: "deer",5: "dog",
    6: "frog",7: "horse",8: "ship",9: "truck"
}
model_path = "./checkpoints/vit_cifar10_model.pth"
img_path = "./infer/horse.jpg"
predicted_class = CIFAR10_CLASSES[predict(img_path, model_path)]
print(predicted_class)