# from llm import Gemini
# gemini = Gemini()
# gemini.get_llm()
# llm = gemini.llm
# llm_cb = gemini.llm_cb
# response = llm.invoke("hello")
# print(response,llm_cb)

from qr_validator import QRValidation
from gcp_data import DataUpload
from wallet_upload import WalletUpload
from vertex_ai import VertexAIRetrieval

# validation = QRValidation()
# qr_mapping = validation.execute_task()

# gcp_loader = DataUpload(qr_mapping)
# url_mapping = gcp_loader.execute_task()

# uploader = WalletUpload(url_mapping)
# wallet_links = uploader.execute_task()

retriever = VertexAIRetrieval()
response = retriever.execute_task("what is my recent internet bill")
print(response)

# from llm import Gemini
# gemini = Gemini()
# gemini.get_llm()
# gemini.llm.invoke("hello")







