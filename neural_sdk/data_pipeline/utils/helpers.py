"""
Kalshi WebSocket Infrastructure - Helper Utilities
"""

from typing import List, Any, Generator


def batch_list(items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Split a list into batches
    
    Args:
        items: List to batch
        batch_size: Size of each batch
    
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def convert_price_from_centicents(centi_cents: int, precision: int = 4) -> float:
    """
    Convert price from centi-cents to dollars
    
    Args:
        centi_cents: Price in centi-cents (1/10000 of a dollar)
        precision: Number of decimal places
    
    Returns:
        Price in dollars
    """
    return round(centi_cents / 10000, precision)