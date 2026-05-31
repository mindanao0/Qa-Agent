#!/usr/bin/env python3
"""Probe CDP Accessibility.getFullAXTree structure."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from playwright.async_api import async_playwright
from collections import Counter


async def probe():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://demo.playwright.dev/todomvc", timeout=30000)
        await page.wait_for_load_state("networkidle", timeout=15000)

        cdp = await page.context.new_cdp_session(page)
        try:
            await cdp.send("Accessibility.enable")
            result = await cdp.send("Accessibility.getFullAXTree")
        finally:
            await cdp.detach()

        nodes = result.get("nodes", [])
        print(f"Total CDP nodes: {len(nodes)}")

        # Show first 10 nodes structure
        print("\nFirst 10 nodes:")
        for n in nodes[:10]:
            role_val = (n.get("role") or {}).get("value", "?")
            name_val = (n.get("name") or {}).get("value", "")
            ignored = n.get("ignored", False)
            parent = n.get("parentId", "NONE")
            children = n.get("childIds", [])
            print(f"  nodeId={n.get('nodeId')} role={role_val!r} name={name_val!r} ignored={ignored} parent={parent!r} nChildren={len(children)}")

        # Count by role
        role_counts = Counter((n.get("role") or {}).get("value", "unknown") for n in nodes)
        print("\nRole distribution (top 10):", role_counts.most_common(10))

        # Count ignored
        ignored_count = sum(1 for n in nodes if n.get("ignored", False))
        print(f"Ignored nodes: {ignored_count}/{len(nodes)}")

        # Find root (no parentId in map)
        node_map = {n.get("nodeId"): n for n in nodes if n.get("nodeId")}
        roots = [n for n in nodes if n.get("parentId") not in node_map]
        print(f"\nRoot candidates: {len(roots)}")
        for r in roots[:3]:
            role_val = (r.get("role") or {}).get("value", "?")
            name_val = (r.get("name") or {}).get("value", "")
            print(f"  nodeId={r.get('nodeId')} role={role_val!r} name={name_val!r} ignored={r.get('ignored')}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(probe())
