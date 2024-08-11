import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.model import PConvUNet  # Adjust the import based on your directory structure

# Define paths
input_img_path = "examples/occlusion1.png"
output_img_path = "examples/occlusion2_pconv_output.png"

# Load and preprocess the image
occluded_img = Image.open(input_img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust this size based on your model's input requirements
    transforms.ToTensor(),
])
img_tensor = transform(occluded_img).unsqueeze(0)  # Add batch dimension

# Create a binary mask (assuming central occlusion, adjust as needed)
mask_tensor = torch.ones_like(img_tensor)
mask_tensor[:, :, 128:-128, :][:, :, :, 128:-128] = 0

# Initialize the model
model = PConvUNet(finetune=False)  # Set finetune=True if you're in fine-tuning mode
model.eval()

# Load pre-trained weights
checkpoint = torch.load('pretrained_pconv.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])

# Run the model for inpainting
with torch.no_grad():
    output_img, _ = model(img_tensor, mask_tensor)

# Convert the output tensor to an image
output_img = output_img.squeeze(0).cpu()
output_pil = transforms.ToPILImage()(output_img)

# Save and display the output
output_pil.save(output_img_path)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Occluded Input")
plt.imshow(occluded_img)
plt.subplot(1, 2, 2)
plt.title("Completed Output")
plt.imshow(output_pil)
plt.show()
