"""
Vercel Serverless Entry Point
Routes all requests to the FastAPI application.
"""

import sys
import os

# Add the backend root to Python path so `app.*` and `ml.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402, F401

# Vercel expects a `handler` or an ASGI app named `app`
# FastAPI (ASGI) is auto-detected by @vercel/python
