import cv2
import uuid
import qrcode
from PIL import Image
from pyzbar.pyzbar import decode
import re
import os
from const import input_dir , output_dir

class QRValidation:
    def is_uuid(self, text):
        uuid_regex = re.compile(
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$', re.IGNORECASE)
        return bool(uuid_regex.match(text))

    def detect_qr_or_barcode(self, image_path):
        img = cv2.imread(image_path)
        detected_codes = decode(img)
        contents = [code.data.decode("utf-8") for code in detected_codes]
        return contents

    def generate_qr(self, content, qr_path):
        qr = qrcode.QRCode(box_size=10, border=2)
        qr.add_data(content)
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color="black", back_color="white")
        img_qr.save(qr_path)
        return qr

    def attach_qr_to_image(self, original_image_path, qr_path, output_path, position='bottom_right'):
        original = Image.open(original_image_path).convert("RGB")
        qr = Image.open(qr_path).convert("RGB")

        qr_width = original.width // 5
        qr = qr.resize((qr_width, qr_width))

        if position == 'bottom_right':
            position = (original.width - qr.width - 10, original.height - qr.height - 10)

        original.paste(qr, position)
        original.save(output_path)

    def process_receipt(self, image_path, output_path):
        contents = self.detect_qr_or_barcode(image_path)
        # print(f"{os.path.basename(image_path)}: Detected codes:", contents)

        for content in contents:
            if self.is_uuid(content):
                print("Valid UUID found. Skipping QR generation.")
                return  # Exit if valid UUID found

        new_uuid = str(uuid.uuid4())
        # print("Generating new UUID:", new_uuid)
        qr_temp_path = "temp_qr.png"
        qr = self.generate_qr(new_uuid, qr_temp_path)
        self.attach_qr_to_image(image_path, qr_temp_path, output_path)
        os.remove(qr_temp_path)
        return qr

    def execute_task(self):
        os.makedirs(output_dir, exist_ok=True)
        qr_mapping = {}

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_image_path = os.path.join(input_dir, filename)
                output_image_path = os.path.join(output_dir, filename)
                qr = self.process_receipt(input_image_path, output_image_path)
                qr_mapping[filename] = qr
        return qr_mapping



