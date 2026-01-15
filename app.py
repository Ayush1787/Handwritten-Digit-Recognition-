import torch
import base64
import config
import matplotlib
import numpy as np
from PIL import Image
from io import BytesIO
from train import MnistModel
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify

matplotlib.use('Agg')

MODEL = None
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

# ------------------ Hook class ------------------
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

def register_hook():
    save_output = SaveOutput()
    for layer in MODEL.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            layer.register_forward_hook(save_output)
    return save_output

def module_output_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

# ------------------ Graph helpers ------------------
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

def prob_img(probs):
    fig, ax = plt.subplots()
    rects = ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 110)
    ax.set_title('Probability % of Digit by Model')
    autolabel(rects, ax)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def interpretability_img(save_output):
    images = module_output_to_numpy(save_output.outputs[0])

    fig, _ = plt.subplots(figsize=(20, 20))
    plt.suptitle("Interpretability by Model", fontsize=40)

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[0, i])
        plt.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ------------------ Prediction ------------------
def mnist_prediction(img):
    save_output = register_hook()
    img = img.to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(x=img)

    probs = torch.exp(outputs)[0] * 100
    probencoded = prob_img(probs)
    interpretencoded = interpretability_img(save_output)

    pred = torch.argmax(outputs, 1).item()
    save_output.clear()

    return pred, probencoded, interpretencoded

# ------------------ ROUTES ------------------

# ✅ HOME PAGE (THIS FIXES 404)
@app.route("/", methods=["GET"])
def home():
    return render_template("default.html")

# ✅ PROCESS ROUTE
@app.route("/process", methods=["POST"])
def process():
    data_url = request.get_data().decode('utf-8')
    img_base64 = data_url.split(',')[1]
    img_bytes = base64.b64decode(img_base64)

    img = Image.open(BytesIO(img_bytes)).convert('L')
    img = img.resize((28, 28))

    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    digit, probencoded, interpretencoded = mnist_prediction(img)

    return jsonify({
        'data': str(digit),
        'probencoded': probencoded,
        'interpretencoded': interpretencoded
    })

# ------------------ MAIN ------------------
if __name__ == "__main__":
    MODEL = MnistModel(classes=10)
    MODEL.load_state_dict(torch.load("checkpoint/mnist.pt", map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)
