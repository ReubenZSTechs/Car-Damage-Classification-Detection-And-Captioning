import torch
import numpy as np
from PIL import Image

from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from torchvision import models, transforms


class Resnet50CustomModel(torch.nn.Module):
    def __init__(self, dropout1=0.3, dropout2=0.4, dropout3=0.5, out1=1024, out2=512, out3=256):
        super().__init__()

        self.extractor = models.resnet50(pretrained=True)
        in_features = self.extractor.fc.in_features # Extract only the vector feature importance
        self.extractor.fc = torch.nn.Identity()
        
        self.mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout1),

            torch.nn.Linear(in_features=out1, out_features=out2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout2),

            torch.nn.Linear(in_features=out2, out_features=out3),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout3),
        )
        
        self.classifier_head = torch.nn.Linear(in_features=out3, out_features=2)

    def forward(self, x):
        img_features = self.extractor(x)
        mlp_features = self.mlp_layer(img_features)
        prediction = self.classifier_head(mlp_features)

        return prediction


class DamageInferenceModel:
    def __init__(
        self,
        device: str,
        resnet_pth_path = 'models/inference_model/resnet_damage.pth',
        yolo_location_path = 'models/inference_model/yolo_location.pt',
        yolo_severity_path = 'models/inference_model/yolo_severity.pt',
        captioning_path = 'models/inference_model/caption_model',
        damage_threshold: float = 0.3,
    ):
        super().__init__()

        self.device = device
        self.damage_threshold = damage_threshold
        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

         # -------------------------
        # RESNET — DAMAGE CLASSIFIER
        # -------------------------
        checkpoint = torch.load(resnet_pth_path, map_location=device)

        self.resnet = Resnet50CustomModel(**checkpoint["config"])
        self.resnet.load_state_dict(checkpoint["state_dict"])
        self.resnet.to(device).eval()

        # -------------------------
        # YOLO MODELS
        # -------------------------
        self.yolo_location = YOLO(yolo_location_path)
        self.yolo_severity = YOLO(yolo_severity_path)

        # -------------------------
        # ViT + GPT2 CAPTIONING
        # -------------------------
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(
            captioning_path
        ).to(device).eval()

        self.caption_processor = ViTImageProcessor.from_pretrained(captioning_path)
        self.caption_tokenizer = AutoTokenizer.from_pretrained(captioning_path)

    # --------------------------------------------------
    # DAMAGE CLASSIFICATION
    # --------------------------------------------------
    @torch.no_grad()
    def _predict_damage(self, image: Image.Image):
        x = self.resnet_transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.resnet(x), dim=1)
        damaged_prob = probs[0, 1].item()
        return damaged_prob > self.damage_threshold, damaged_prob

    # --------------------------------------------------
    # YOLO HELPER
    # --------------------------------------------------
    @staticmethod
    def _extract_best_detection(result):
        if result.boxes is None or len(result.boxes) == 0:
            return None, None
        idx = result.boxes.conf.argmax()
        return (
            result.boxes.xyxy[idx].cpu().numpy(),
            int(result.boxes.cls[idx])
        )

    # --------------------------------------------------
    # SEVERITY
    # --------------------------------------------------
    @torch.no_grad()
    def _predict_severity(self, image: Image.Image):
        result = self.yolo_severity(image)[0]
        box, cls = self._extract_best_detection(result)
        if cls is None:
            return "unknown", None
        return self.yolo_severity.names[cls], box

    # --------------------------------------------------
    # INFERENCE ENTRY POINT
    # --------------------------------------------------
    @torch.no_grad()
    def __call__(self, image: Image.Image):
        image = image.convert("RGB")
        np_image = np.array(image)

        # DAMAGE CLASSIFICATION
        is_damaged, damage_prob = self._predict_damage(image)

        # CAPTIONING
        pixel_values = self.caption_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        output_ids = self.caption_model.generate(
            pixel_values, max_length=50, num_beams=5
        )

        raw_caption = self.caption_tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        if not is_damaged:
            return {
                "image_np": np_image,
                "damaged": False,
                "damage_probability": damage_prob,
                "caption": f"A car with no visible damage. {raw_caption}",
                'severity': False
            }

        # LOCATION
        loc_result = self.yolo_location(image)[0]
        loc_box, loc_cls = self._extract_best_detection(loc_result)
        location = (
            self.yolo_location.names[loc_cls]
            if loc_cls is not None else "unknown"
        )

        # SEVERITY
        severity, severity_box = self._predict_severity(image)

        return {
            "image_np": np_image,
            "damaged": True,
            "damage_probability": damage_prob,
            "location": location,
            "severity": severity,
            "location_box": loc_box,
            "severity_box": severity_box,
            "caption": f"A car with {location} damage of {severity} severity. {raw_caption}",
        }