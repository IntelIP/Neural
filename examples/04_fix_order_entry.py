import argparse
import asyncio
import os
import uuid

from dotenv import load_dotenv

from neural.trading import FIXConnectionConfig, KalshiFIXClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interact with the Kalshi FIX order entry gateway."
    )
    parser.add_argument("--symbol", required=True, help="Market ticker to trade (FIX tag 55).")
    parser.add_argument(
        "--side",
        choices=["buy", "sell", "yes", "no"],
        default="buy",
        help="Order side (buy=yes / sell=no).",
    )
    parser.add_argument("--quantity", type=int, default=1, help="Contracts to trade (tag 38).")
    parser.add_argument("--price", type=int, required=True, help="Limit price in cents (tag 44).")
    parser.add_argument("--host", default="fix.elections.kalshi.com", help="FIX gateway host.")
    parser.add_argument("--port", type=int, default=8228, help="FIX gateway port.")
    parser.add_argument(
        "--target", default="KalshiNR", help="TargetCompID for the chosen endpoint."
    )
    parser.add_argument(
        "--sender",
        help="SenderCompID / FIX API key (defaults to KALSHI_FIX_API_KEY or KALSHI_API_KEY_ID).",
    )
    parser.add_argument("--heartbeat", type=int, default=30, help="Heartbeat interval in seconds.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send a NewOrderSingle after login. Without this flag we only establish the session.",
    )
    parser.add_argument(
        "--cancel-after",
        type=int,
        default=0,
        help="If >0, submit an OrderCancelRequest after N seconds.",
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="How long to keep the session open before logout."
    )
    return parser


def handle_message(message) -> None:
    parsed = KalshiFIXClient.to_dict(message)
    print(
        {
            tag: value
            for tag, value in parsed.items()
            if tag in (35, 11, 17, 37, 39, 150, 58, 10, 198, 434, 102, 103, 380)
        }
    )


async def run(args) -> None:
    sender = args.sender or os.getenv("KALSHI_FIX_API_KEY") or os.getenv("KALSHI_API_KEY_ID")
    config = FIXConnectionConfig(
        host=args.host,
        port=args.port,
        target_comp_id=args.target,
        sender_comp_id=sender,
        heartbeat_interval=args.heartbeat,
    )

    if not config.sender_comp_id:
        raise SystemExit("SenderCompID is required. Set --sender or KALSHI_FIX_API_KEY.")

    async with KalshiFIXClient(config=config, on_message=handle_message) as fix:
        if args.execute:
            cl_ord_id = str(uuid.uuid4())
            await fix.new_order_single(
                cl_order_id=cl_ord_id,
                symbol=args.symbol,
                side=args.side,
                quantity=args.quantity,
                price=args.price,
            )
            print(f"Submitted order {cl_ord_id}")

            if args.cancel_after > 0:
                await asyncio.sleep(args.cancel_after)
                cancel_id = str(uuid.uuid4())
                await fix.cancel_order(
                    cl_order_id=cancel_id,
                    orig_cl_order_id=cl_ord_id,
                    symbol=args.symbol,
                    side=args.side,
                )
                print(f"Submitted cancel {cancel_id}")

        await asyncio.sleep(args.duration)


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
