"""Dashboard utilities."""

try:
    from .static_dashboard import write_dashboard_html
except ImportError:
    from static_dashboard import write_dashboard_html

__all__ = ["write_dashboard_html"]
