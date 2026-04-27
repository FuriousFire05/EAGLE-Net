from torchvision.datasets import EuroSAT
from torchvision import transforms
import matplotlib.pyplot as plt

# Transform (VERY important)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset
dataset = EuroSAT(
    root="data/raw",
    download=True,
    transform=transform
)

# Print basic info
print("Total samples:", len(dataset))
print("Classes:", dataset.classes)

# Show 1 sample image
image, label = dataset[0]

print("Label index:", label)
print("Class name:", dataset.classes[label])

# Convert tensor to image for display
img = image.permute(1, 2, 0)

plt.imshow(img)
plt.title(dataset.classes[label])
plt.axis("off")
plt.show()
