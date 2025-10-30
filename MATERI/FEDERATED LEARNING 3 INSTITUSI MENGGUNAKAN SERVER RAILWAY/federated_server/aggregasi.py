import requests

SERVER_URL = "https://federatedserver-production.up.railway.app"

print("📡 Mengirim permintaan agregasi FedAvg ke server...")
response = requests.post(f"{SERVER_URL}/aggregate")

print("\n Respons server:")
print(response.json())





