1. Run gcp_data.py for uploading receipts to google cloud storage and getting a url.
2. Run wallet_upload.py to upload images to wallet by passing the urls to the google wallet api and create associated unique QRs and a google pass for each receipt.
3. Run retrieve.py to retrieve relevant chunks based on user query leveraging Vertex AI.
4. Run agents.py to access the lang graph swarm and interact with multiple sub-agents (tools).