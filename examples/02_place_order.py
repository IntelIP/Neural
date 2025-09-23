import argparse
import uuid
from typing import Any

from dotenv import load_dotenv

from neural.trading import TradingClient


def pick_default_ticker(client: TradingClient) -> str:
	markets = client.markets.get_markets(limit=1, status="open") or {}
	items = markets.get("markets") or []
	if not items:
		raise RuntimeError("No open markets returned; specify --ticker explicitly")
	return items[0]["ticker"]


def main() -> None:
	load_dotenv()
	parser = argparse.ArgumentParser(description="Submit a Kalshi limit order via the Neural trading client.")
	parser.add_argument("--ticker", help="Market ticker to trade.")
	parser.add_argument("--side", choices=["yes", "no"], default="yes", help="Contract side to trade (YES buys vs NO sells).")
	parser.add_argument("--action", choices=["buy", "sell"], default="buy", help="Portfolio action to perform.")
	parser.add_argument("--count", type=int, default=1, help="Number of contracts.")
	parser.add_argument("--price", type=int, help="Limit price in cents (1-99). Required for limit orders.")
	parser.add_argument("--execute", action="store_true", help="Actually send the order. Otherwise run in dry-run mode.")
	args = parser.parse_args()

	with TradingClient() as client:
		selected_ticker = args.ticker or pick_default_ticker(client)
		order_request: dict[str, Any] = {
			"ticker": selected_ticker,
			"side": args.side,
			"action": args.action,
			"count": args.count,
			"type": "limit",
			"client_order_id": str(uuid.uuid4()),
			"yes_price": args.price if args.side == "yes" else None,
			"no_price": args.price if args.side == "no" else None,
		}
		if not args.price:
			raise SystemExit("--price is required to build the limit order payload")

		print(f"Using ticker: {selected_ticker}")
		print(f"Account balance: {client.portfolio.get_balance()}")
		print(f"Dry-run payload: {order_request}")

		if not args.execute:
			print("Pass --execute to submit the order against production.")
			return

		response = client.portfolio.create_order(**order_request)
		print("Order accepted:", response)


if __name__ == "__main__":
	main()
