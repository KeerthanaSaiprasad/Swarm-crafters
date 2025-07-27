import uuid
import json
import requests
from typing import Dict
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from const import (
    ISSUER_ID, GENERIC_CLASSES_URL, GENERIC_OBJECTS_URL,
    CLASS_ID, SERVICE_ACCOUNT_FILE, SCOPES, SAVE
)


class WalletUpload:
    def __init__(self, url_mapping: Dict):
        self.url_mapping = url_mapping
        self.issuer_id = ISSUER_ID
        self.class_id = CLASS_ID
        self.credentials = None
        self.headers = None
        self._authenticate()

    def _authenticate(self):
        self.credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        self.credentials.refresh(Request())
        self.headers = {
            'Authorization': f'Bearer {self.credentials.token}',
            'Content-Type': 'application/json'
        }

    def create_class(self):
        class_payload = {
            "id": f"{self.issuer_id}.{self.class_id}",
            "issuerName": "My Store Receipts",
            "reviewStatus": "UNDER_REVIEW"
        }

        response = requests.post(GENERIC_CLASSES_URL, headers=self.headers, data=json.dumps(class_payload))

        if response.status_code == 200 or "already exists" in response.text:
            print("✅ Class created or already exists.")
        else:
            print("❌ Class creation failed:", response.text)

    def create_object(self, image_url, qr):
        object_id = f"{self.issuer_id}.receipt_object_{qr}"

        object_payload = {
            "id": object_id,
            "classId": f"{self.issuer_id}.{self.class_id}",
            "genericType": "GENERIC_TYPE_UNSPECIFIED",
            "cardTitle": {
                "defaultValue": {
                    "language": "en-US",
                    "value": "Digital Receipt"
                }
            },
            "header": {
                "defaultValue": {
                    "language": "en-US",
                    "value": "Receipt Image"
                }
            },
            "subheader": {
                "defaultValue": {
                    "language": "en-US",
                    "value": "Issued by Merchant"
                }
            },
            "heroImage": {
                "sourceUri": {
                    "uri": image_url,
                    "description": "Uploaded receipt image"
                }
            },
            "barcode": {
                "type": "QR_CODE",
                "value": f"receipt-{object_id}"
            }
        }

        response = requests.post(GENERIC_OBJECTS_URL, headers=self.headers, data=json.dumps(object_payload))

        if response.status_code == 200:
            print(f"✅ Object created for image: {image_url}")
            return object_id
        else:
            print(f"❌ Object creation failed for {image_url}: {response.text}")
            return None

    def get_save_url(self, object_id):
        return f"{SAVE}/{object_id}"

    def execute_task(self):
        self.create_class()
        save_urls = []

        for image_url, qr in self.url_mapping.items():
            # print(f"\n🔄 Processing image: {image_url}")
            object_id = self.create_object(image_url, qr)
            if object_id:
                save_url = self.get_save_url(object_id)
                self.retrieve_object(object_id)
                save_urls.append(save_url)

        return save_urls
