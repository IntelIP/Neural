from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from typing import Any

from neural.deployment import create_provider, list_providers

CLI_COMMANDS = [
    "doctor",
    "capabilities",
    "providers list",
    "markets list",
    "quote",
    "positions",
    "paper order",
    "deployments list",
    "deployments status",
    "deployments logs",
    "deployments stop",
]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    json_output = bool(args.json)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    try:
        payload = args.handler(args)
        _emit_success(payload, json_output=json_output, formatter=getattr(args, "formatter", None))
        return 0
    except Exception as exc:  # pragma: no cover - exercised via CLI entrypoint
        _emit_error(exc, json_output=json_output)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural", description="Neural SDK command line interface")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser("doctor", help="Inspect local Neural environment health")
    doctor_parser.set_defaults(handler=_handle_doctor, formatter=_format_doctor)

    capabilities_parser = subparsers.add_parser(
        "capabilities", help="Describe CLI, provider, and paper-trading capabilities"
    )
    capabilities_parser.set_defaults(handler=_handle_capabilities, formatter=_format_capabilities)

    providers_parser = subparsers.add_parser("providers", help="Inspect deployment providers")
    providers_subparsers = providers_parser.add_subparsers(dest="providers_command")
    providers_list_parser = providers_subparsers.add_parser("list", help="List discovered providers")
    providers_list_parser.set_defaults(handler=_handle_providers_list, formatter=_format_providers)

    markets_parser = subparsers.add_parser("markets", help="Query Neural market data")
    markets_subparsers = markets_parser.add_subparsers(dest="markets_command")
    markets_list_parser = markets_subparsers.add_parser("list", help="List Kalshi markets")
    markets_list_parser.add_argument("--limit", type=int, default=10)
    markets_list_parser.add_argument("--category")
    markets_list_parser.add_argument("--status", default="open")
    markets_list_parser.set_defaults(handler=_handle_markets_list, formatter=_format_markets)

    quote_parser = subparsers.add_parser("quote", help="Fetch a market quote")
    quote_parser.add_argument("market_id")
    quote_parser.set_defaults(handler=_handle_quote, formatter=_format_quote)

    positions_parser = subparsers.add_parser("positions", help="Fetch account positions")
    positions_parser.set_defaults(handler=_handle_positions, formatter=_format_positions)

    paper_parser = subparsers.add_parser("paper", help="Execute paper-trading commands")
    paper_subparsers = paper_parser.add_subparsers(dest="paper_command")
    paper_order_parser = paper_subparsers.add_parser("order", help="Place a paper order")
    paper_order_parser.add_argument("--market-id", required=True)
    paper_order_parser.add_argument("--side", required=True, choices=["yes", "no"])
    paper_order_parser.add_argument("--quantity", required=True, type=int)
    paper_order_parser.add_argument("--order-type", default="market", choices=["market", "limit"])
    paper_order_parser.add_argument("--price", type=float)
    paper_order_parser.add_argument("--initial-capital", type=float, default=10000.0)
    paper_order_parser.add_argument("--data-dir", default="paper_trading_data")
    paper_order_parser.set_defaults(handler=_handle_paper_order, formatter=_format_paper_order)

    deployments_parser = subparsers.add_parser("deployments", help="Inspect runtime deployments")
    deployments_parser.add_argument("--provider", default=os.getenv("NEURAL_DEPLOYMENT_PROVIDER"))
    deployments_parser.add_argument("--workspace-name", default=os.getenv("NEURAL_DAYTONA_WORKSPACE"))
    deployments_parser.add_argument("--runner-image", default=os.getenv("NEURAL_DAYTONA_IMAGE"))
    deployments_parser.add_argument("--project-name", default="neural")
    deployments_parser.add_argument("--environment", default="paper")
    deployments_parser.add_argument("--daytona-binary", default=os.getenv("NEURAL_DAYTONA_BINARY"))
    deployments_subparsers = deployments_parser.add_subparsers(dest="deployments_command")

    deployments_list_parser = deployments_subparsers.add_parser("list", help="List deployments")
    deployments_list_parser.set_defaults(handler=_handle_deployments_list, formatter=_format_deployments)

    deployments_status_parser = deployments_subparsers.add_parser("status", help="Get deployment status")
    deployments_status_parser.add_argument("deployment_id")
    deployments_status_parser.set_defaults(
        handler=_handle_deployments_status, formatter=_format_deployment_status
    )

    deployments_logs_parser = deployments_subparsers.add_parser("logs", help="Fetch deployment logs")
    deployments_logs_parser.add_argument("deployment_id")
    deployments_logs_parser.add_argument("--tail", type=int, default=50)
    deployments_logs_parser.set_defaults(
        handler=_handle_deployments_logs, formatter=_format_deployment_logs
    )

    deployments_stop_parser = deployments_subparsers.add_parser("stop", help="Stop a deployment")
    deployments_stop_parser.add_argument("deployment_id")
    deployments_stop_parser.set_defaults(
        handler=_handle_deployments_stop, formatter=_format_deployment_stop
    )

    return parser


def _handle_doctor(_: argparse.Namespace) -> dict[str, Any]:
    providers = _safe_list_providers()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "credentials": {
            "kalshi_api_key_id": bool(os.getenv("KALSHI_API_KEY_ID")),
            "kalshi_private_key_base64": bool(os.getenv("KALSHI_PRIVATE_KEY_BASE64")),
            "kalshi_private_key_path": bool(os.getenv("KALSHI_PRIVATE_KEY_PATH")),
            "kalshi_env": os.getenv("KALSHI_ENV", "prod"),
        },
        "tooling": {
            "uv_available": shutil.which("uv") is not None,
            "bun_available": shutil.which("bun") is not None,
            "daytona_cli_available": shutil.which("daytona") is not None,
            "installed_modules": {
                "kalshi_python": _module_available("kalshi_python"),
                "textblob": _module_available("textblob"),
                "vaderSentiment": _module_available("vaderSentiment"),
                "daytona": _module_available("daytona"),
                "docker": _module_available("docker"),
            },
        },
        "providers": providers,
        "commands": CLI_COMMANDS,
    }


def _handle_capabilities(_: argparse.Namespace) -> dict[str, Any]:
    providers = _safe_list_providers()
    return {
        "cli": {
            "commands": CLI_COMMANDS,
            "json_envelope": {
                "success": {"ok": True, "data": "..."},
                "error": {"ok": False, "error": {"code": "...", "message": "..."}},
            },
        },
        "trading": {
            "paper_first": True,
            "live_trading": False,
            "market_read": True,
            "positions_read": True,
            "paper_order": True,
        },
        "platform": {
            "providers": providers,
            "private_provider_installed": "daytona" in providers,
            "daytona_sdk_available": _module_available("daytona"),
            "daytona_cli_available": shutil.which("daytona") is not None,
            "gated_modes": ["deployments", "runtime-logs"],
            "deployment_controls": ["list", "status", "logs", "stop"],
        },
    }


def _handle_providers_list(_: argparse.Namespace) -> dict[str, Any]:
    providers = _safe_list_providers()
    return {"providers": providers, "count": len(providers)}


def _handle_markets_list(args: argparse.Namespace) -> dict[str, Any]:
    KalshiHTTPClient = _get_kalshi_http_client_class()

    params: dict[str, Any] = {"limit": args.limit, "status": args.status}
    if args.category:
        params["category"] = args.category

    with KalshiHTTPClient() as client:
        payload = client.get("/markets", params=params)

    rows = payload.get("markets", [])
    normalized = [
        {
            "ticker": row.get("ticker"),
            "title": row.get("title") or row.get("name"),
            "status": row.get("status"),
            "yes_ask": row.get("yes_ask"),
            "no_ask": row.get("no_ask"),
            "volume": row.get("volume"),
        }
        for row in rows
        if isinstance(row, dict)
    ]
    return {"markets": normalized, "count": len(normalized)}


def _handle_quote(args: argparse.Namespace) -> dict[str, Any]:
    KalshiHTTPClient = _get_kalshi_http_client_class()

    with KalshiHTTPClient() as client:
        payload = client.get(f"/markets/{args.market_id}")

    market = payload.get("market", payload)
    return {
        "market_id": market.get("ticker") or market.get("id") or args.market_id,
        "title": market.get("title") or market.get("name"),
        "yes_bid": market.get("yes_bid"),
        "yes_ask": market.get("yes_ask"),
        "no_bid": market.get("no_bid"),
        "no_ask": market.get("no_ask"),
        "last_price": market.get("last_price") or market.get("yes_price"),
        "volume": market.get("volume"),
        "status": market.get("status"),
        "raw": market,
    }


def _handle_positions(_: argparse.Namespace) -> dict[str, Any]:
    KalshiHTTPClient = _get_kalshi_http_client_class()

    with KalshiHTTPClient() as client:
        payload = client.get("/portfolio/positions")

    positions = payload.get("positions", payload if isinstance(payload, list) else [])
    normalized = [
        {
            "market_id": row.get("market_id") or row.get("ticker"),
            "side": row.get("side"),
            "quantity": row.get("quantity") or row.get("count"),
            "entry_price": row.get("avg_price") or row.get("entry_price"),
            "current_price": row.get("current_price"),
            "unrealized_pnl": row.get("unrealized_pnl"),
        }
        for row in positions
        if isinstance(row, dict)
    ]
    return {"positions": normalized, "count": len(normalized)}


def _handle_paper_order(args: argparse.Namespace) -> dict[str, Any]:
    PaperTradingClient = _get_paper_trading_client_class()

    client = PaperTradingClient(
        initial_capital=args.initial_capital,
        save_trades=False,
        data_dir=args.data_dir,
    )
    if args.price is not None:
        client.update_market_price(args.market_id, args.side, args.price)

    result = asyncio.run(
        client.place_order(
            market_id=args.market_id,
            side=args.side,
            quantity=args.quantity,
            order_type=args.order_type,
            price=args.price,
        )
    )
    return {"order": _serialize(result), "portfolio": client.get_portfolio()}


def _handle_deployments_list(args: argparse.Namespace) -> dict[str, Any]:
    provider_name, provider = _build_provider(args)
    data = asyncio.run(provider.list_deployments())
    return {
        "provider": provider_name,
        "deployments": [_serialize(item) for item in data],
        "count": len(data),
    }


def _handle_deployments_status(args: argparse.Namespace) -> dict[str, Any]:
    provider_name, provider = _build_provider(args)
    status = asyncio.run(provider.status(args.deployment_id))
    return {"provider": provider_name, "deployment": _serialize(status)}


def _handle_deployments_logs(args: argparse.Namespace) -> dict[str, Any]:
    provider_name, provider = _build_provider(args)
    logs = asyncio.run(provider.logs(args.deployment_id, tail=args.tail))
    return {"provider": provider_name, "deployment_id": args.deployment_id, "logs": logs}


def _handle_deployments_stop(args: argparse.Namespace) -> dict[str, Any]:
    provider_name, provider = _build_provider(args)
    stopped = asyncio.run(provider.stop(args.deployment_id))
    return {"provider": provider_name, "deployment_id": args.deployment_id, "stopped": bool(stopped)}


def _build_provider(args: argparse.Namespace) -> tuple[str, Any]:
    provider_name = _resolve_provider_name(getattr(args, "provider", None))
    kwargs: dict[str, Any] = {}
    for source, dest in (
        ("workspace_name", "workspace_name"),
        ("runner_image", "runner_image"),
        ("project_name", "project_name"),
        ("environment", "environment"),
        ("daytona_binary", "daytona_binary"),
    ):
        value = getattr(args, source, None)
        if value:
            kwargs[dest] = value
    return provider_name, create_provider(provider_name, **kwargs)


def _resolve_provider_name(explicit_provider: str | None) -> str:
    if explicit_provider:
        return explicit_provider

    providers = _safe_list_providers()
    if "daytona" in providers:
        return "daytona"
    if "docker" in providers:
        return "docker"
    if len(providers) == 1:
        return providers[0]
    if not providers:
        raise RuntimeError(
            "No deployment providers discovered. Install a provider plugin or pass --provider explicitly."
        )
    raise RuntimeError(
        "Multiple deployment providers discovered. Pass --provider explicitly: "
        + ", ".join(providers)
    )


def _safe_list_providers() -> list[str]:
    try:
        return list_providers()
    except Exception:
        return []


def _get_kalshi_http_client_class() -> Any:
    from neural.auth.http_client import KalshiHTTPClient

    return KalshiHTTPClient


def _get_paper_trading_client_class() -> Any:
    from neural.trading.paper_client import PaperTradingClient

    return PaperTradingClient


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if hasattr(value, "model_dump"):
        return _serialize(value.model_dump())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _emit_success(payload: dict[str, Any], *, json_output: bool, formatter: Any) -> None:
    if json_output:
        print(json.dumps({"ok": True, "data": _serialize(payload)}, indent=2, sort_keys=True))
        return
    if formatter:
        print(formatter(payload))
        return
    print(payload)


def _emit_error(exc: Exception, *, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": exc.__class__.__name__,
                        "message": str(exc),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(f"error: {exc}", file=sys.stderr)


def _format_doctor(payload: dict[str, Any]) -> str:
    installed = payload["tooling"]["installed_modules"]
    providers = ", ".join(payload["providers"]) or "(none)"
    return "\n".join(
        [
            f"Neural doctor @ {payload['timestamp']}",
            f"Python: {payload['python']['version']} ({payload['python']['implementation']})",
            "Credentials:",
            f"  Kalshi API key present: {payload['credentials']['kalshi_api_key_id']}",
            f"  Private key present: {payload['credentials']['kalshi_private_key_base64'] or payload['credentials']['kalshi_private_key_path']}",
            "Tooling:",
            f"  uv={payload['tooling']['uv_available']} bun={payload['tooling']['bun_available']} daytona_cli={payload['tooling']['daytona_cli_available']}",
            "Optional modules:",
            f"  kalshi_python={installed['kalshi_python']} textblob={installed['textblob']} vaderSentiment={installed['vaderSentiment']} daytona={installed['daytona']}",
            f"Providers: {providers}",
        ]
    )


def _format_capabilities(payload: dict[str, Any]) -> str:
    providers = ", ".join(payload["platform"]["providers"]) or "(none)"
    return "\n".join(
        [
            "Neural capabilities",
            f"CLI commands: {', '.join(payload['cli']['commands'])}",
            f"Paper-first trading: {payload['trading']['paper_first']}",
            f"Private provider installed: {payload['platform']['private_provider_installed']}",
            f"Daytona SDK available: {payload['platform']['daytona_sdk_available']}",
            f"Daytona CLI available: {payload['platform']['daytona_cli_available']}",
            f"Providers: {providers}",
        ]
    )


def _format_providers(payload: dict[str, Any]) -> str:
    providers = payload["providers"]
    if not providers:
        return "No deployment providers discovered."
    return "Deployment providers:\n" + "\n".join(f"  - {provider}" for provider in providers)


def _format_markets(payload: dict[str, Any]) -> str:
    lines = [f"Markets ({payload['count']}):"]
    for market in payload["markets"]:
        lines.append(
            f"  - {market['ticker']}: {market['title']} | yes_ask={market['yes_ask']} no_ask={market['no_ask']}"
        )
    return "\n".join(lines)


def _format_quote(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"{payload['market_id']}: {payload['title']}",
            f"yes bid/ask: {payload['yes_bid']} / {payload['yes_ask']}",
            f"no bid/ask: {payload['no_bid']} / {payload['no_ask']}",
            f"last price: {payload['last_price']} volume: {payload['volume']}",
        ]
    )


def _format_positions(payload: dict[str, Any]) -> str:
    if not payload["positions"]:
        return "No positions found."
    lines = [f"Positions ({payload['count']}):"]
    for position in payload["positions"]:
        lines.append(
            f"  - {position['market_id']} [{position['side']}]: qty={position['quantity']} unrealized_pnl={position['unrealized_pnl']}"
        )
    return "\n".join(lines)


def _format_paper_order(payload: dict[str, Any]) -> str:
    order = payload["order"]
    return "\n".join(
        [
            f"Paper order success: {order.get('success')}",
            f"Order ID: {order.get('order_id')}",
            f"Message: {order.get('message')}",
            f"Portfolio value: {payload['portfolio'].get('portfolio_value')}",
        ]
    )


def _format_deployments(payload: dict[str, Any]) -> str:
    lines = [f"{payload['provider']} deployments ({payload['count']}):"]
    for item in payload["deployments"]:
        lines.append(
            f"  - {item.get('deployment_id')}: {item.get('status')} ({item.get('environment')})"
        )
    return "\n".join(lines)


def _format_deployment_status(payload: dict[str, Any]) -> str:
    deployment = payload["deployment"]
    return "\n".join(
        [
            f"{deployment.get('deployment_id')}: {deployment.get('status')}",
            f"health: {deployment.get('health_status')}",
            f"uptime: {deployment.get('uptime_seconds')}",
        ]
    )


def _format_deployment_logs(payload: dict[str, Any]) -> str:
    lines = [f"Logs for {payload['deployment_id']}:"]
    lines.extend(f"  {line}" for line in payload["logs"])
    return "\n".join(lines)


def _format_deployment_stop(payload: dict[str, Any]) -> str:
    return f"Stopped {payload['deployment_id']}: {payload['stopped']}"

