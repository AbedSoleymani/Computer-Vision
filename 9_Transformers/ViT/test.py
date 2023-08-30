import numpy as np
from PIL import Image, ImageDraw
import torch

top_guesses = 5

imagenet_labels = dict(enumerate(open("./9_Transformers/ViT/imagenet_labels.txt")))

model = torch.load("./9_Transformers/ViT/saved_models/model.pth")
model.eval()

img_name = "dog2.jpg"  # "cat.jpg", "dog.jpg", "dog2.jpg", "school_bus.jpg"
img = Image.open("./9_Transformers/ViT/imgs/"+img_name)
img = img.resize((384, 384))
img_draw = ImageDraw.Draw(img)
input = (np.array(img) / 128) - 1  # in the range -1, 1
inp = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_probs, top_ixs = probs[0].topk(top_guesses)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i+1}: {cls[:30]:<30} --- {prob:.4f}")
    img_draw.text((15,15+10*i), f"{i+1}: {cls[:30]:30} --- {prob:.4f}", (0, 0, 0))

img.show()
img.save("./9_Transformers/ViT/imgs/Annotated_"+img_name)