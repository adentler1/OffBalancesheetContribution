#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 08:33:22 2025

@author: alexander
"""

import sys
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Optional


# ----------------------------- Logging setup -----------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "football_fetcher.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("football")
# ---------------------- Graceful Exit / Error Handling --------------------
class GracefulExit(Exception):
    """Raised to stop the pipeline cleanly with a clear reason."""


@dataclass
class ExitContext:
    reason: str
    detail: Optional[str] = None
    stage: Optional[str] = None
    requests_remaining: Optional[int] = None


class GracefulExitManager:
    """
    Wraps a run sequence, records the last error/exit reason,
    ensures progress is saved, and emits consistent messages.
    """
    def __init__(self):
        self.ctx: Optional[ExitContext] = None

    def exit(self, reason: str, detail: Optional[str] = None,
             stage: Optional[str] = None, requests_remaining: Optional[int] = None):
        self.ctx = ExitContext(reason=reason, detail=detail, stage=stage,
                               requests_remaining=requests_remaining)
        raise GracefulExit(reason)

    def handle(self, save_progress_callable=None):
        """
        Use around your top-level run. Example:
            try:
                ... run pipeline ...
            except Exception:
                exit_mgr.handle(progress_tracker.save_now)
        """
        etype, value, tb = sys.exc_info()
        msg = f"{etype.__name__}: {value}"
        stack = "".join(traceback.format_tb(tb)) if tb else ""
        LOGGER.error("Unhandled exception:\n%s\n%s", msg, stack)

        # Ensure progress is saved if caller provides a callable
        if save_progress_callable:
            try:
                save_progress_callable()
            except Exception as e:
                LOGGER.error("Failed to save progress while handling exception: %s", e)

        # Emit a clean stdout message (short + useful)
        print("⚠️  Aborted cleanly.")
        if self.ctx:
            r = self.ctx
            print(f"Reason: {r.reason}")
            if r.stage:
                print(f"Stage: {r.stage}")
            if r.detail:
                print(f"Detail: {r.detail}")
            if r.requests_remaining is not None:
                print(f"Requests remaining (last known): {r.requests_remaining}")
        else:
            print("Reason: unexpected error (see log).")