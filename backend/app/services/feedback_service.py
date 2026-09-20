# backend/app/services/feedback_service.py
"""
Phase‑I FeedbackService
职责：
1. 校验FeedbackCreate入参（使用review_result_id/review_defect_id外键，不再直接传task_id/defect_id）
2. 写入feedback_item表（严格使用ORM真实字段，禁止task_id/defect_id直接构造ORM对象）
3. 落库后分发到RuleEvolutionService：
   - false_negative / new_rule_candidate → 生成rule_candidate草稿
   - knowledge_gap → 记录知识缺口
   - correct_defect / false_positive / suggestion_update：仅落库，不生成候选
⚠️ 重要说明：
feedback_item表**没有task_id、defect_id列**；
task_id来自review_result.task_id（外键review_result_id）
defect_id来自review_defect.defect_id（外键review_defect_id）
comment_by(字符串用户名) → ORM created_by 存users.id(int主键)
"""
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.schemas.feedback import FeedbackCreate, FeedbackOut, FeedbackType, SuggestionDiff
from app.persistence.models import FeedbackItem, ReviewResult, ReviewDefect
from app.services.rule_evolution_service import RuleEvolutionService


class FeedbackService:
    def __init__(self):
        self.rule_evolution = RuleEvolutionService()

    def submit_feedback(self, db: Session, feedback_in: FeedbackCreate) -> FeedbackOut:
        """
        提交专家反馈：落库feedback_item，触发rule_evolution后置处理
        """
        # 1. 校验关联外键是否存在
        review_result = db.scalar(select(ReviewResult).where(ReviewResult.id == feedback_in.review_result_id))
        review_defect = db.scalar(select(ReviewDefect).where(ReviewDefect.id == feedback_in.review_defect_id))
        if not review_result:
            raise ValueError(f"review_result id={feedback_in.review_result_id} 不存在")
        if not review_defect:
            raise ValueError(f"review_defect id={feedback_in.review_defect_id} 不存在")

        # 2. 构造ORM记录：只使用FeedbackItem真实存在数据库字段！！
        db_feedback = FeedbackItem(
            review_result_id=feedback_in.review_result_id,
            review_defect_id=feedback_in.review_defect_id,
            feedback_type=feedback_in.feedback_type.value,
            comment=feedback_in.expert_suggestion,
            expert_suggestion=feedback_in.expert_suggestion,
            suggestion_adopted=feedback_in.suggestion_adopted,
            suggestion_diff_json=[d.model_dump() for d in feedback_in.suggestion_diff]
            if feedback_in.suggestion_diff else [],
            attached_refs=[],
            created_by=feedback_in.created_by,
        )
        db.add(db_feedback)
        db.flush()
        feedback_id = db_feedback.id

        candidate_ref: Optional[str] = None

        # 3. 根据feedback_type分发规则演进逻辑
        if feedback_in.feedback_type in (FeedbackType.FALSE_NEGATIVE, FeedbackType.NEW_RULE_CANDIDATE):
            candidate_ref = self.rule_evolution.generate_candidate_from_feedback(
                db=db,
                feedback_id=feedback_id,
                case_id=None,
                hint_payload=feedback_in.rule_candidate
            )
            db_feedback.rule_candidate_ref = candidate_ref
            db.flush()

        # knowledge_gap 不生成rule_candidate，知识缺口导出由rule_evolution导出接口处理
        db.commit()
        db.refresh(db_feedback)

        # 4. 组装输出DTO：JOIN回填业务视图字段 task_id / defect_id
        out = FeedbackOut(
            id=db_feedback.id,
            review_result_id=db_feedback.review_result_id,
            review_defect_id=db_feedback.review_defect_id,
            task_id=review_result.task_id,
            defect_id=review_defect.defect_id,
            feedback_type=FeedbackType(db_feedback.feedback_type),
            expert_suggestion=db_feedback.expert_suggestion,
            suggestion_diff=[SuggestionDiff(**x) for x in db_feedback.suggestion_diff_json],
            rule_candidate_ref=db_feedback.rule_candidate_ref,
            created_by=db_feedback.created_by,
            created_at=db_feedback.created_at.isoformat() if db_feedback.created_at else ""
        )
        return out


if __name__ == "__main__":
    """CLI简易入口，供demo脚本调用"""
    import sys
    import json
    from app.persistence.db import get_db_session

    service = FeedbackService()
    if len(sys.argv) < 2:
        print("usage: python -m app.services.feedback_service <feedback_json_path>")
        sys.exit(1)
    json_path = sys.argv[1]
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    fb_in = FeedbackCreate(**raw)
    with get_db_session() as db:
        res = service.submit_feedback(db, fb_in)
    print(res.model_dump_json(indent=2))
