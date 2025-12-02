import gradio as gr
import torch
import numpy as np
from PIL import Image
import torchvision
import model  # 你的 Zero-DCE model.py

# --- 加载模型 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

DCE = model.enhance_net_nopool().to(device)
DCE.load_state_dict(torch.load("snapshots/Epoch99_old.pth", map_location=device))
DCE.eval()

def enhance(img):
    # 将 PIL 图像 → tensor
    img_np = np.asarray(img) / 255.0
    img_tensor = torch.from_numpy(img_np).float().permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        _, enhanced, _ = DCE(img_tensor)

    enhanced = enhanced.squeeze(0).cpu()
    enhanced = (enhanced * 255).clamp(0,255).byte().permute(1,2,0).numpy()

    return Image.fromarray(enhanced)

# --- Gradio UI ---
demo = gr.Interface(
    fn=enhance,
    inputs=gr.Image(type="pil", label="上传一张低光图"),
    outputs=gr.Image(type="pil", label="增强结果"),
    title="Zero-DCE 低光增强",
    description="上传一张图片，模型将自动进行亮度增强。",
)

demo.launch(server_name="10.13.3.242")
