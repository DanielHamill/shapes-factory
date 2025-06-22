# -*- coding: utf-8 -*-
# https://github.com/nickjj/docker-flask-example/blob/main/config/gunicorn.py

# import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
# workers = int(os.getenv("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2))
workers = int(os.getenv("WEB_CONCURRENCY", 1))
threads = int(os.getenv("PYTHON_MAX_THREADS", 1))
timeout = int(os.getenv("WEB_TIMEOUT", 120))