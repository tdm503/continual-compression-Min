import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt  # For displaying the image if output is an image tensor
from cloc import get_model

# 1. Load pre-trained model
model = get_model('msh_vr', pretrained=True)
model.eval()

# 2. Load và xử lý ảnh
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize đúng 256x256
        transforms.ToTensor(),          # [0,1] chuẩn (bỏ normalize [-1,1])
    ])

    img_tensor = transform(img)  # (C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # Thêm batch dimension: (1, C, H, W)
    return img_tensor


# 3. Hàm inference
def inference(image_path):
    input_tensor = load_image(image_path)  # (1, 3, 256, 256)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    print("Inference thành công! Output type:", type(output))

    # Kiểm tra nếu output là OrderedDict, trích xuất tensor từ đó
    if isinstance(output, dict):  # Đảm bảo rằng output là dict (OrderedDict)
        print("Các keys trong output:", output.keys())  # In ra các keys có trong OrderedDict
        
        # Trích xuất và in các chỉ số (loss, bpp, mse, psnr, ...)
        print("Loss:", output['loss'].item())
        
        # Bỏ gọi .item() cho các giá trị float như 'bpp', 'mse', 'psnr'
        print("BPP:", output['bpp'])  
        print("MSE:", output['mse'])
        print("PSNR:", output['psnr'])
        
    return output

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py [path_to_image]")
        exit(1)
    
    image_path = sys.argv[1]
    inference(image_path)
