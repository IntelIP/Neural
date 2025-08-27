"""Utility functions for formatting data."""

from datetime import datetime, timedelta
from typing import Any, Optional, Union
from decimal import Decimal


def format_currency(value: Union[float, Decimal, int], 
                   decimals: int = 2, 
                   include_sign: bool = True) -> str:
    """Format a value as currency.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        include_sign: Include + sign for positive values
        
    Returns:
        Formatted currency string
    """
    if value is None:
        return "$0.00"
    
    value = float(value)
    
    if include_sign and value > 0:
        return f"+${value:,.{decimals}f}"
    elif value < 0:
        return f"-${abs(value):,.{decimals}f}"
    else:
        return f"${value:,.{decimals}f}"


def format_percentage(value: Union[float, Decimal], 
                     decimals: int = 2, 
                     include_sign: bool = True) -> str:
    """Format a value as percentage.
    
    Args:
        value: Numeric value to format (already in percentage)
        decimals: Number of decimal places
        include_sign: Include + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "0.00%"
    
    value = float(value)
    
    if include_sign and value > 0:
        return f"+{value:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}%"


def format_number(value: Union[float, int, Decimal], 
                 decimals: int = 0,
                 use_thousands_separator: bool = True) -> str:
    """Format a number with optional thousands separator.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        use_thousands_separator: Use comma as thousands separator
        
    Returns:
        Formatted number string
    """
    if value is None:
        return "0"
    
    value = float(value)
    
    if use_thousands_separator:
        return f"{value:,.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"


def format_timestamp(timestamp: Union[datetime, str], 
                    format_type: str = "full") -> str:
    """Format a timestamp for display.
    
    Args:
        timestamp: Datetime object or ISO string
        format_type: Format type ('full', 'date', 'time', 'relative')
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        return "N/A"
    
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    
    if format_type == "full":
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "date":
        return timestamp.strftime("%Y-%m-%d")
    elif format_type == "time":
        return timestamp.strftime("%H:%M:%S")
    elif format_type == "relative":
        return format_relative_time(timestamp)
    else:
        return str(timestamp)


def format_relative_time(timestamp: datetime) -> str:
    """Format timestamp as relative time (e.g., '5 minutes ago').
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Relative time string
    """
    now = datetime.utcnow()
    diff = now - timestamp
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"


def format_market_ticker(ticker: str) -> str:
    """Format market ticker for display.
    
    Args:
        ticker: Raw market ticker
        
    Returns:
        Formatted ticker string
    """
    if not ticker:
        return "N/A"
    
    # Truncate long tickers
    if len(ticker) > 20:
        return ticker[:17] + "..."
    
    return ticker.upper()


def format_trade_side(side: str) -> str:
    """Format trade side with emoji indicator.
    
    Args:
        side: Trade side ('buy', 'sell', 'yes', 'no')
        
    Returns:
        Formatted side string
    """
    side_lower = side.lower() if side else ""
    
    if side_lower in ['buy', 'yes']:
        return f"📈 {side.upper()}"
    elif side_lower in ['sell', 'no']:
        return f"📉 {side.upper()}"
    else:
        return side.upper() if side else "N/A"


def format_status(status: str) -> str:
    """Format status with emoji indicator.
    
    Args:
        status: Status string
        
    Returns:
        Formatted status string
    """
    status_lower = status.lower() if status else ""
    
    status_emojis = {
        'running': '🟢',
        'stopped': '🔴',
        'paused': '🟡',
        'error': '❌',
        'pending': '⏳',
        'filled': '✅',
        'cancelled': '❌',
        'failed': '❌'
    }
    
    emoji = status_emojis.get(status_lower, '⚫')
    return f"{emoji} {status.upper()}"


def format_metric_delta(current: float, previous: float) -> tuple:
    """Format metric delta for display.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Tuple of (delta_value, delta_percentage, is_positive)
    """
    if previous == 0:
        delta = current
        delta_pct = 100 if current > 0 else -100 if current < 0 else 0
    else:
        delta = current - previous
        delta_pct = (delta / abs(previous)) * 100
    
    is_positive = delta >= 0
    
    return (delta, delta_pct, is_positive)


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h"


def color_for_value(value: float, 
                   positive_color: str = "#00ff88",
                   negative_color: str = "#ff4444",
                   neutral_color: str = "#ffffff") -> str:
    """Get color based on value sign.
    
    Args:
        value: Numeric value
        positive_color: Color for positive values
        negative_color: Color for negative values
        neutral_color: Color for zero/neutral values
        
    Returns:
        Color string
    """
    if value > 0:
        return positive_color
    elif value < 0:
        return negative_color
    else:
        return neutral_color