import os
from typing import Dict
from google.cloud import storage
from google.api_core import exceptions
from const import LOCAL_INVOICE_IMAGE_PATH

class DataUpload:

    def __init__(self, qr_mapping: Dict):
        self.qr_mapping = qr_mapping

    def upload_and_make_public(self, source_file_path, bucket_name):
        try:
            # Use the filename as destination blob name
            destination_blob_name = os.path.basename(source_file_path)

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            blob.upload_from_filename(source_file_path)

            # Make public
            blob.make_public()
            return blob.public_url

        except FileNotFoundError:
            # print(f"Error: File {source_file_path} not found.")
            return None
        except exceptions.GoogleAPIError as e:
            # print(f"GCS API error: {e}")
            return None
        except Exception as e:
            # print(f"Unexpected error: {e}")
            return None

    def execute_task(self):
        url_mapping = {}
        bucket_name = os.getenv("BUCKETNAME")
        image_dir = LOCAL_INVOICE_IMAGE_PATH  # Ensure this is a directory path
        public_urls = []

        if not os.path.exists(image_dir):
            # print(f"Error: Image directory {image_dir} does not exist.")
            return []

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(image_dir, filename)
                public_url = self.upload_and_make_public(file_path, bucket_name)
                if public_url:
                    public_urls.append(public_url)
                url_mapping[public_url] = self.qr_mapping[filename]
                
        return url_mapping
