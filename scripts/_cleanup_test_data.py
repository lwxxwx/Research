# scripts/_cleanup_test_data.py
from sqlalchemy import text
from app.persistence.db import engine

with engine.begin() as conn:
    r1 = conn.execute(text("DELETE FROM review_defect WHERE defect_id LIKE 'DEF-MOCK-E2E-%'"))
    r2 = conn.execute(text("DELETE FROM review_result WHERE task_id LIKE 'TEST-%'"))
    print(f"cleaned review_defect: {r1.rowcount}, review_result: {r2.rowcount}")