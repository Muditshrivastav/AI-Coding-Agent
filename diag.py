"""Quick diagnostic to identify harness initialization failure."""
import asyncio
import os
import traceback

from dotenv import load_dotenv

load_dotenv()


async def diagnose():
    db_dsn = (
        os.getenv("supabase_uri")
        or os.getenv("SUPABASE_URI")
        or os.getenv("DATABASE_URL")
        or os.getenv("SUPABASE_DB_URL")
    )
    print(f"DB DSN configured: {bool(db_dsn)}")

    # Try LangGraphRuntime.create
    from agent.session_manager import SessionManager
    from frameworks.langgraph_runtime import LangGraphRuntime

    sessions = SessionManager(root_dir=".")
    print("SessionManager loaded OK")
    try:
        runtime = await LangGraphRuntime.create(
            session_manager=sessions, root_dir="."
        )
        print(f"LangGraphRuntime OK: checkpointer={type(runtime.checkpointer).__name__}")
        await runtime.aclose()
    except Exception as exc:
        print(f"LangGraphRuntime.create FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()


asyncio.run(diagnose())
