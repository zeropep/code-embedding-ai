"""Formatting utility functions"""


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format

    Examples:
        >>> format_duration(0.19866275787353516)
        '0h 0m 0.20s'
        >>> format_duration(125.5)
        '0h 2m 5.50s'
    """
    if seconds < 0:
        return "0h 0m 0.00s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours}h {minutes}m {secs:.2f}s"
