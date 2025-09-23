import os
from dotenv import load_dotenv
from neural.auth.env import get_api_key_id, get_private_key_material
from neural.auth import AuthClient, KalshiSigner

def main():
	load_dotenv()
	api_key_id = get_api_key_id()
	priv_pem = get_private_key_material()

	signer = KalshiSigner(api_key_id, priv_pem)
	client = AuthClient(signer, env=os.getenv("KALSHI_ENV"))

	resp = client.get("/trade-api/v2/portfolio/balance")
	print(resp)

if __name__ == "__main__":
	main()

