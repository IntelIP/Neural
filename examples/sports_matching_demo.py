"""Offline MLB proposition/rule comparison using explicitly synthetic identities.

Run from the repository: uv run python examples/sports_matching_demo.py
No network, credentials, orders, fee assumptions or settlement simulation.
"""

import json
from pathlib import Path

from neural.sports import SportsMarket, compare_sports_markets


def main() -> None:
    fixture = json.loads(Path(__file__).with_name("sports-matching-fixtures.json").read_text())
    print(fixture["notice"])
    for case in fixture["cases"]:
        left = SportsMarket.from_dict(fixture["markets"][case["left"]])
        right = SportsMarket.from_dict(
            {**fixture["markets"][case["right"]], **case["right_overrides"]}
        )
        result = compare_sports_markets(left, right)
        assert result.status == case["expected"], case["name"]
        assert result.proposition == case["expected_proposition"], case["name"]
        print(json.dumps({"case": case["name"], **result.to_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
