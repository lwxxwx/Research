"""
app/persistence/db.py
数据库会话工厂
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from app.core.config import settings

engine = create_engine(settings.database_url, echo=settings.database_echo)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Session:
    """获取数据库会话，上下文管理器，自动commit/rollback/close"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
