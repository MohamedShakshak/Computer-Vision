from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

# تحميل نموذج تصنيف الصور ومعالج الصور
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# استخدام GPU إذا كان متاحًا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_image(image_path):
    # تحميل الصورة وتحويلها إلى صيغة RGB إذا لزم الأمر
    image = Image.open(image_path).convert("RGB")
    
    # تحضير البيانات باستخدام المعالج
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # التنبؤ بالتصنيف
    with torch.no_grad():
        outputs = model(**inputs)
    
    # الحصول على الفئة المتوقعة
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    return label

# تحديد مسار الصورة على جهازك
image_path = "D://images//image.jpg"

# إجراء التصنيف
predicted_label = classify_image(image_path)
print(f"Predicted Label: {predicted_label}")
