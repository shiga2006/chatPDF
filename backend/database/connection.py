import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine.url import make_url
from backend.config import settings

logger = logging.getLogger(__name__)

try:
    import pymysql  # type: ignore
except Exception:
    pymysql = None

def _build_sqlite_url() -> str:
    sqlite_path = Path(settings.CHROMA_DIR).parent / "chatpdf.db"
    return f"sqlite:///{sqlite_path.as_posix()}"

# Extract connection parameters to pre-create the DB if it does not exist
try:
    db_url = make_url(settings.DATABASE_URL)
    if db_url.drivername.startswith("mysql") and pymysql is not None:
        host = db_url.host or "localhost"
        port = db_url.port or 3306
        user = db_url.username or "root"
        password = db_url.password or ""
        database = db_url.database

        # Establish raw connection to MySQL server to ensure DB exists
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset='utf8mb4'
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            conn.commit()
            logger.info(f"Database '{database}' verified/created successfully.")
        except Exception as e:
            logger.warning(f"Could not verify/create database via raw connection: {e}")
        finally:
            conn.close()
    else:
        logger.warning("MySQL/PyMySQL unavailable; using local SQLite fallback for development.")
except Exception as e:
    logger.error(f"Error parsing database URL or pre-creating database: {e}")

# Create engine and session factories
database_url = settings.DATABASE_URL
engine_kwargs = {
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}

try:
    if not make_url(database_url).drivername.startswith("mysql"):
        raise ValueError("Non-MySQL URL configured; use SQLite fallback for local development.")
    engine = create_engine(database_url, **engine_kwargs)
except Exception as e:
    logger.warning(f"Falling back to SQLite database because the configured DB could not be initialized: {e}")
    database_url = _build_sqlite_url()
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
        **engine_kwargs
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency generator for database sessions in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
