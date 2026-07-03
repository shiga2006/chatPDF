import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine.url import make_url
import pymysql
from backend.config import settings

logger = logging.getLogger(__name__)

# Extract connection parameters to pre-create the DB if it does not exist
try:
    db_url = make_url(settings.DATABASE_URL)
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
except Exception as e:
    logger.error(f"Error parsing database URL or pre-creating database: {e}")

# Create engine and session factories
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
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
