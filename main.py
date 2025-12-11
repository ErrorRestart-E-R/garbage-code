"""
AI VTuber Discord Bot (entrypoint)

이 파일은 엔트리포인트만 유지합니다.
- Discord bot wiring: app/bot.py
- Conversation orchestration: app/controller.py
- VTS lipsync service: services/lipsync.py
"""

import sys
import config
from app.bot import run_bot

if __name__ == "__main__":
    from all_api_testing import run_all_tests

    print("\nStarting AI VTuber Bot...\n")
    if config.ENABLE_PREFLIGHT_CHECKS:
        if not run_all_tests():
            print("\n[FAIL] Pre-flight checks failed.\n")
            sys.exit(1)
        print("[OK] All systems operational\n")
    else:
        print("[WARN] Pre-flight checks skipped (set ENABLE_PREFLIGHT_CHECKS=True in config.py to enable)\n")
    
    run_bot()
