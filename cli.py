"""
cli.py - Command-line interface for the Autonomous Coding Agent Harness.
"""

import asyncio
import argparse
import uuid
from agent.core import CodingAgentHarness


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Coding Agent Harness CLI")
    parser.add_argument("prompt", type=str, help="Software task description to execute")
    parser.add_argument("--thread-id", type=str, default=None, help="Thread ID for session/checkpoints")
    parser.add_argument("--root-dir", type=str, default=".", help="Root directory for sandbox")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run")
    parser.add_argument("--approve", action="store_true", help="Approve paused action when resuming")

    args = parser.parse_args()
    thread_id = args.thread_id or str(uuid.uuid4())

    harness = CodingAgentHarness(root_dir=args.root_dir)

    if args.resume:
        print(f"Resuming thread {thread_id} (Approved={args.approve})...")
        res = await harness.resume(thread_id, approved=args.approve)
    else:
        print(f"Starting execution for thread {thread_id}...")
        res = await harness.run(args.prompt, thread_id=thread_id)

    print("Result:", res)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
