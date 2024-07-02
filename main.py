import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
import urllib.request
from PIL import Image
from PIL import ImageDraw,ImageFont
import os
transform = transforms.Compose([
    #transforms.Resize(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def Download_img(dir):
    img_urls = []
    img_paths = []
    if not os.path.exists('./pic'):
        url = "https://pic.sogou.com/pics?query=%E8%B6%B3%E7%90%83%E6%AF%94%E8%B5%9B"
        urllib.request.urlretrieve(url,"pic")
    else:
        file = open("pic",'r',encoding='utf-8')
        content = file.read()
        buf = list("123456789")
        for i in range(len(content)):
            buf[i%9] = content[i]
            if buf== list("drag-img="):
                i+=2
                buf2 = ""
                while(content[i]!="\""):
                    buf2+=content[i]
                    i+=1
                img_urls.append(buf2)
        file.close()
        img_urls = img_urls
        for p in range(len(img_urls)):
            img_paths.append(os.path.join(dir,"football"+str(p)+".jpg"))
            urllib.request.urlretrieve(img_urls[p],os.path.join(dir,"football"+str(p)+".jpg"))
    return img_paths
img_paths = Download_img("Picture")
def main():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=True)
    model.eval()
    threshold = 0.8
    font = ImageFont.truetype("arial.ttf", 12)
    for I in img_paths:
        img_pil = Image.open(I)
        img_tensor = transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            predictions = model(img_tensor)
        predictions = {k: v.cpu().numpy() for k, v in predictions[0].items()}
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']
        indices = scores > threshold
        Draw = ImageDraw.Draw(img_pil)
        for box, score, label in zip(boxes[indices], scores[indices], labels[indices]):
            box = box.tolist()
            Draw.rectangle(box, outline='red')
            Draw.text((box[0], box[1] - 20), f"{weights.meta['categories'][label]}: {score:.2f}", fill='white',
                      font=font)
        img_pil.save("./Predict/"+I)
if __name__ == "__main__":
    main()

