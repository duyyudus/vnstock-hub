import asyncio
import os
import re
from pathlib import Path
from typing import AsyncGenerator

import pytest
import asyncpg
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from httpx import AsyncClient, ASGITransport


def _bootstrap_test_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env.test"
    if not env_path.exists():
        raise RuntimeError(f"Missing required test environment file: {env_path}")

    os.environ["APP_ENV_FILE"] = str(env_path)
    load_dotenv(dotenv_path=env_path, override=True)


_bootstrap_test_env()

from app.db.database import Base, get_db, async_session as global_async_session
from app.core.config import settings
from app.main import app


async def ensure_db_exists(database_url: str):
    """
    Ensures the test database exists by connecting to the 'postgres' database
    and creating it if it's missing.
    """
    # Parse URL: postgresql+asyncpg://user:pass@host:port/dbname
    pattern = r"postgresql(?:\+asyncpg)?://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/([^/]+)"
    match = re.match(pattern, database_url)
    if not match:
        return

    user, password, host, port, db_name = match.groups()
    port = int(port) if port else 5432

    try:
        # Connect to 'postgres' to run CREATE DATABASE
        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres"
        )
        try:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                print(f"\n[test_setup] Created test database: {db_name}")
        finally:
            await conn.close()
    except Exception as e:
        print(f"\n[test_setup] Info: Database existence check skipped (already exists or permission issue): {e}")


_db_initialized = False


@pytest.fixture(scope="function")
async def test_engine():
    """
    Create an engine for each test to avoid loop affinity issues.
    Initializes the schema (drop and create) once per test session.
    """
    global _db_initialized
    database_url = settings.database_url

    # Create engine for this test's loop
    engine = create_async_engine(database_url, echo=False)

    if not _db_initialized:
        await ensure_db_exists(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        _db_initialized = True
    
    yield engine
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Function-scoped session that runs in a transaction and rolls back.
    """
    async with test_engine.connect() as connection:
        transaction = await connection.begin()
        
        # Configure global session to use this connection specifically
        global_async_session.configure(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint"
        )
        
        session = global_async_session()
        
        # Override FastAPI dependency for the duration of the test
        app.dependency_overrides[get_db] = lambda: session
        
        yield session

        await session.close()
        if transaction.is_active:
            await transaction.rollback()
        
        app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client for testing API endpoints.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
