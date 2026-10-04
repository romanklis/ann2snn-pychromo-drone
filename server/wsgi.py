"""WSGI entrypoint for gunicorn: ``gunicorn -b 0.0.0.0:8080 server.wsgi:application``."""

from server.app import app as application

__all__ = ["application"]
