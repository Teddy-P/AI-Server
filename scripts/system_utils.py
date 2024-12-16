import os
from datetime import datetime

def get_current_time():
    """
    Returns the current system time in HH:MM:SS format.
    """
    return datetime.now().strftime("%H:%M:%S")

def get_current_date():
    """
    Returns the current system date in YYYY-MM-DD format.
    """
    return datetime.now().strftime("%Y-%m-%d")

def get_system_uptime():
    """
    Returns the system uptime (Linux only).
    """
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    except FileNotFoundError:
        return "System uptime information is not available."

def get_disk_usage(path="/"):
    """
    Returns disk usage stats for the given path.
    """
    usage = os.statvfs(path)
    total = usage.f_blocks * usage.f_frsize
    free = usage.f_bfree * usage.f_frsize
    used = total - free
    return {
        "total": total,
        "used": used,
        "free": free,
        "percent_used": (used / total) * 100 if total else 0
    }
