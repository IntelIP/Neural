import argparse
import contextlib
import json
import signal
import threading
import time

from dotenv import load_dotenv

from neural.trading import KalshiWebSocketClient


def main() -> None:
	load_dotenv()
	parser = argparse.ArgumentParser(description="Subscribe to Kalshi websocket channels and stream updates.")
	parser.add_argument("--ticker", required=True, help="Market ticker to monitor.")
	parser.add_argument("--channel", default="orderbook_delta", help="Channel to subscribe to (orderbook_delta, trades, positions, etc.)")
	parser.add_argument("--duration", type=int, default=60, help="How long to stream in seconds.")
	args = parser.parse_args()

	stop_event = threading.Event()
	subscription_ref: dict[str, int | None] = {"sid": None}

	def handle_message(message: dict) -> None:
		if message.get("type") == "subscribed" and message.get("sid"):
			subscription_ref["sid"] = message["sid"]
		print(json.dumps(message, separators=(",", ":")))

	with KalshiWebSocketClient(on_message=handle_message) as client:
		client.subscribe([args.channel], params={"market_tickers": [args.ticker]})

		def shutdown(signum, frame):
			stop_event.set()
			sid = subscription_ref.get("sid")
			if sid is not None:
				with contextlib.suppress(Exception):
					client.unsubscribe([sid])

		signal.signal(signal.SIGINT, shutdown)
		signal.signal(signal.SIGTERM, shutdown)

		end_time = time.time() + args.duration
		while time.time() < end_time and not stop_event.is_set():
			time.sleep(0.5)

		sid = subscription_ref.get("sid")
		if sid is not None:
			client.unsubscribe([sid])


if __name__ == "__main__":
	main()
