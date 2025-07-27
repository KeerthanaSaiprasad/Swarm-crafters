input_dir = r"C:/Users/youkn/Downloads/hack5/src/invoice_images"
output_dir = r"C:/Users/youkn/Downloads/hack5/src/qr_images"

LOCAL_INVOICE_IMAGE_PATH = r"C:/Users/youkn/Downloads/hack5/src/qr_images"
DESTINATION_BLOB_NAME = None


GENERIC_CLASSES_URL = 'https://walletobjects.googleapis.com/walletobjects/v1/genericClass  ' # <-- Trailing spaces
GENERIC_OBJECTS_URL = 'https://walletobjects.googleapis.com/walletobjects/v1/genericObject  ' # <-- Trailing spaces

ISSUER_ID = '3388000000022957641'
CLASS_SUFFIX = 'receipt_class1'
CLASS_ID = f"{ISSUER_ID}.{CLASS_SUFFIX}"

SERVICE_ACCOUNT_FILE = "C:/Users/youkn/Downloads/hack5/local/swarmcrafters-467117-7085cb7342f8.json"
SCOPES = ['https://www.googleapis.com/auth/wallet_object.issuer']

SAVE = 'https://pay.google.com/gp/v/save/'