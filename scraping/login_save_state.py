"""
Save RoyaleAPI login state for reuse by new_scrape.py.
Run once, log in manually (and pass any Cloudflare challenge), then press Enter.
State is written to scraping/myGoogleAuth.json (same path the scraper uses).
"""
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

# ─── Config (must match new_scrape.py) ─────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
STORAGE_STATE_PATH = ROOT_DIR / "scraping" / "myGoogleAuth.json"
PERSISTENT_PROFILE_DIR = ROOT_DIR / "scraping" / "chrome_profile"
GOTO_URL = "https://royaleapi.com/"
GOTO_TIMEOUT_MS = 60_000

# Launch options that reduce automation signals without breaking the browser
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
]


async def _apply_stealth_if_available(context):
    """Apply playwright_stealth when installed; otherwise no-op."""
    try:
        from playwright_stealth import Stealth
        stealth = Stealth()
        if hasattr(stealth, "apply_stealth_async"):
            await stealth.apply_stealth_async(context)
        elif hasattr(stealth, "apply_async"):
            await stealth.apply_async(context)
    except ImportError:
        pass


async def login_with_regular_browser():
    """Use a normal Chromium context. Good default."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=LAUNCH_ARGS)
        context = await browser.new_context(
            viewport={"width": 1366, "height": 768},
            locale="en-US",
        )
        await _apply_stealth_if_available(context)
        page = await context.new_page()
        try:
            await _do_flow(page, context, browser)
        finally:
            await context.close()
            await browser.close()


async def login_with_persistent_chrome():
    """Use a persistent Chrome profile (real Chrome if installed). Often bypasses Cloudflare better."""
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            str(PERSISTENT_PROFILE_DIR),
            headless=False,
            channel="chrome",
            viewport={"width": 1366, "height": 768},
            locale="en-US",
        )
        page = context.pages[0] if context.pages else await context.new_page()
        try:
            await _do_flow(page, context, None)
        finally:
            await context.close()


async def _do_flow(page, context, browser):
    STORAGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\nNavigating to RoyaleAPI...")
    await page.goto(GOTO_URL, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)

    print("\n" + "─" * 60)
    print("  1. In the browser: sign in (e.g. Google) and pass any Cloudflare check.")
    print("  2. When you see your profile/avatar, you're logged in.")
    print("  3. Return here and press ENTER to save the session.")
    print("─" * 60 + "\n")

    input("Press Enter once fully logged in... ")

    await context.storage_state(path=STORAGE_STATE_PATH)
    size_kb = STORAGE_STATE_PATH.stat().st_size / 1024
    print(f"\nSession saved → {STORAGE_STATE_PATH}")
    print(f"Size: {size_kb:.1f} KB. Use this file in new_scrape.py (it already points here).\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Save RoyaleAPI login state for the scraper.")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Use a persistent Chrome profile (often better against Cloudflare).",
    )
    args = parser.parse_args()

    if args.persistent:
        asyncio.run(login_with_persistent_chrome())
    else:
        asyncio.run(login_with_regular_browser())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
