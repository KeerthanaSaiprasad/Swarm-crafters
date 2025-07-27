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

class WalletUtils: 
    def retrieve_object(self, object_id):
        get_url = f"{GENERIC_OBJECTS_URL}/{object_id}"
        response = requests.get(get_url, headers=self.headers)

        if response.status_code == 200:
            print(f"\n📥 Retrieved object {object_id}:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to retrieve object {object_id}: {response.text}")
