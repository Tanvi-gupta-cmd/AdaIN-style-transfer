
import os
import torch

from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from wtforms import FileField, SubmitField, FloatField
from wtforms.validators import InputRequired

from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static",
    "uploads"
)

app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# --------------------------------------------------
# Upload Form
# --------------------------------------------------

class UploadForm(FlaskForm):

    content = FileField(
        "Content Image",
        validators=[InputRequired()]
    )

    style = FileField(
        "Style Image",
        validators=[InputRequired()]
    )

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField(
        "Generate Stylized Image"
    )


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

encoder_path = os.path.join(
    BASE_DIR,
    "vgg_normalised.pth"
)

decoder_path = os.path.join(
    BASE_DIR,
    "experiment",
    "final_exp",
    "decoder_final.pth"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load Models
# --------------------------------------------------

encoder = VGGEncoder(
    encoder_path
).to(device)

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        decoder_path,
        map_location=device
    )
)

encoder.eval()
decoder.eval()


# --------------------------------------------------
# File Validation
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# --------------------------------------------------
# Neural Style Transfer
# --------------------------------------------------

def style_transfer(
    content_image,
    style_image,
    alpha
):

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content = transform(
        content_image
    ).unsqueeze(0).to(device)

    style = transform(
        style_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        content_features = encoder(
            content,
            is_test=True
        )

        style_features = encoder(
            style,
            is_test=True
        )

        normalized_features = (
            adaptive_instance_normalization(
                content_features,
                style_features
            )
        )

        blended_features = (
            alpha * normalized_features
            + (1 - alpha) * content_features
        )

        output = decoder(
            blended_features
        )

    return output


# --------------------------------------------------
# Save Output
# --------------------------------------------------

def save_image(image, path):

    image = image.detach().cpu()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(
        image
    )

    image.save(path)


# --------------------------------------------------
# Main Page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    error = None

    if form.validate_on_submit():

        content_file = form.content.data
        style_file = form.style.data

        if not allowed_file(
            content_file.filename
        ):

            error = "Invalid content image format."

        elif not allowed_file(
            style_file.filename
        ):

            error = "Invalid style image format."

        else:

            try:

                content_filename = secure_filename(
                    content_file.filename
                )

                style_filename = secure_filename(
                    style_file.filename
                )

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                content_file.save(
                    content_path
                )

                style_file.save(
                    style_path
                )

                content_image = Image.open(
                    content_path
                ).convert("RGB")

                style_image = Image.open(
                    style_path
                ).convert("RGB")

                alpha = float(
                    form.alpha.data
                )

                alpha = max(
                    0.0,
                    min(1.0, alpha)
                )

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )

                result_filename = (
                    "stylized_" +
                    content_filename
                )

                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                save_image(
                    stylized_image,
                    result_path
                )

                result_image = result_filename

            except Exception as e:

                error = str(e)

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        error=error
    )


# --------------------------------------------------
# Uploaded Images
# --------------------------------------------------

@app.route("/uploads/<filename>")
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# --------------------------------------------------
# Example Images
# --------------------------------------------------

@app.route("/examples/<path:filename>")
def send_example(filename):
    examples_folder = os.path.join(BASE_DIR, "examples")

    return send_from_directory(
        os.path.join(
            BASE_DIR,
            "examples"
        ),
        filename
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

