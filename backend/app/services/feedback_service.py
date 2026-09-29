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

v1.1 变更（改进点，逻辑与参数保持不变）：
- ★ 改进 1：新增 FeedbackServiceError / FeedbackNotFoundError 异常类
            替换裸 ValueError（保留兼容：仍继承 ValueError，旧调用方 except ValueError 不受影响）
- ★ 改进 2：db.scalar → db.scalar_one_or_none（更严格：多行时抛错，而非静默取首行）
- ★ 改进 3：__init__ 支持依赖注入（rule_evolution 可选传入），默认行为不变
- ★ 改进 4：加 logger，在关键节点打印日志（提交、外键校验失败、规则候选生成、完成）
- ★ 改进 5：新增内部辅助方法 _validate_refs / _build_orm / _dispatch_evolution / _build_out
            拆分 submit_feedback，保持主流程与原逻辑 1:1 对应
- ★ 改进 6：commit 行为加开关 commit: bool = True，默认保持原行为（内部 commit）
            需要外层控制事务时传 commit=False，由调用方负责 commit
- ★ 改进 7：Optional[str] → str | None（Python 3.11+，等价写法）
- ★ 改进 8：__main__ 内的字符串字面量改为 # 注释（原为无操作字符串，行为不变，仅可读性）
- ★ 改进 9：comment=feedback_in.expert_suggestion 保留原逻辑
            但加注释说明当前设计；后续若两字段语义分离，应分别赋值

feedback_in (FeedbackCreate), commit (bool, 默认 True)
    ↓
★ 改进 4：logger.info 入口日志（review_result_id / review_defect_id / feedback_type）
    ↓
[1] _validate_refs(db, feedback_in)
    ├─ ★ 改进 2：db.scalar_one_or_none（多行抛错）
    ├─ 查 ReviewResult（by review_result_id）
    ├─ 查 ReviewDefect（by review_defect_id）
    ├─ ★ 改进 1：不存在 → 抛 FeedbackNotFoundError（继承 ValueError）
    └─ 返回 (review_result, review_defect)
    ↓
[2] _build_orm(feedback_in)
    ├─ ★ 改进 9：comment = expert_suggestion（保留原逻辑，加语义注释）
    ├─ feedback_type.value → 字符串
    ├─ suggestion_diff → [d.model_dump() for d in ...]（None 时 → []）
    ├─ attached_refs = []
    └─ 返回 FeedbackItem（未入库）
    ↓
    db.add(db_feedback)
    db.flush()                     → 拿 feedback_id
    ↓
[3] _dispatch_evolution(db, feedback_id, feedback_in, db_feedback)
    ├─ if feedback_type in (FALSE_NEGATIVE, NEW_RULE_CANDIDATE):
    │   ├─ ★ 改进 4：logger.info「触发规则候选生成」
    │   ├─ candidate_ref = rule_evolution.generate_candidate_from_feedback(...)
    │   ├─ db_feedback.rule_candidate_ref = candidate_ref
    │   └─ db.flush()
    └─ else: candidate_ref = None（跳过）
    ↓
[4] ★ 改进 6：commit 由参数控制
    ├─ commit=True（默认）：db.commit()
    └─ commit=False：db.flush()（由调用方负责 commit / rollback）
    ↓
    db.refresh(db_feedback)        → 拿 DB 默认值（如 created_at）
    ↓
★ 改进 4：logger.info 完成日志（feedback_id / candidate_ref）
    ↓
[5] _build_out(db_feedback, review_result, review_defect)
    ├─ id ← db_feedback.id
    ├─ review_result_id ← db_feedback.review_result_id
    ├─ review_defect_id ← db_feedback.review_defect_id
    ├─ task_id ← review_result.task_id          （JOIN 回填）
    ├─ defect_id ← review_defect.defect_id      （JOIN 回填）
    ├─ feedback_type ← FeedbackType(db 字符串)
    ├─ expert_suggestion ← db_feedback.expert_suggestion
    ├─ suggestion_diff ← [SuggestionDiff(**x) for x in suggestion_diff_json]
    ├─ rule_candidate_ref ← db_feedback.rule_candidate_ref
    ├─ created_by ← db_feedback.created_by
    └─ created_at ← db_feedback.created_at.isoformat() if ... else ""
    ↓
return FeedbackOut

-待解决问题：RuleCandidate 字段与 rule_evolution 期望不一致，字段没有对齐。
"""
from typing import Optional  # noqa: F401  # ★ 改进 7：保留旧 import 兼容，新代码用 X | None

# ★ 改进 4：新增 logging
import logging

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.schemas.feedback import FeedbackCreate, FeedbackOut, FeedbackType, SuggestionDiff
from app.persistence.models import FeedbackItem, ReviewResult, ReviewDefect
from app.services.rule_evolution_service import RuleEvolutionService


# ★ 改进 4：模块级 logger
logger = logging.getLogger(__name__)


# ============================================================
# ★ 改进 1：自定义异常类
# ------------------------------------------------------------
# 设计：
#   - 均继承 ValueError，保证旧调用方 except ValueError 仍生效
#   - 便于按类型精细捕获（except FeedbackNotFoundError）
#   - 便于后续扩展（如加 code、http_status）
# ============================================================
class FeedbackServiceError(ValueError):
    """FeedbackService 基类异常（继承 ValueError，保持兼容）"""


class FeedbackNotFoundError(FeedbackServiceError):
    """关联外键不存在（review_result / review_defect）"""


class FeedbackService:
    # ★ 改进 3：__init__ 支持依赖注入（默认行为不变）
    # ------------------------------------------------------------
    # 原代码：
    #     def __init__(self):
    #         self.rule_evolution = RuleEvolutionService()
    # 说明：
    #     - 原代码每次实例化都 new 一个 RuleEvolutionService
    #     - 新代码允许外部传入（便于 mock、复用、控制生命周期）
    #     - 不传时行为完全一致
    # ------------------------------------------------------------
    def __init__(self, rule_evolution: "RuleEvolutionService | None" = None):
        # --- 原代码（保留，已弃用） ---
        # self.rule_evolution = RuleEvolutionService()
        self.rule_evolution = rule_evolution or RuleEvolutionService()

    def submit_feedback(
        self,
        db: Session,
        feedback_in: FeedbackCreate,
        # ★ 改进 6：commit 行为开关，默认 True 保持原行为
        commit: bool = True,
    ) -> FeedbackOut:
        """
        提交专家反馈：落库feedback_item，触发rule_evolution后置处理

        ★ 改进 6：新增 commit 参数
            - commit=True（默认）：方法内部 db.commit()，与原行为一致
            - commit=False：方法内部只 flush，由调用方负责 commit / rollback
        """
        # ★ 改进 4：入口日志
        logger.info(
            f"submit_feedback 开始：review_result_id={feedback_in.review_result_id}, "
            f"review_defect_id={feedback_in.review_defect_id}, "
            f"feedback_type={feedback_in.feedback_type}"
        )

        # ============================================================
        # 1. 校验关联外键是否存在
        # ============================================================
        # --- 原代码（保留，已弃用） ---
        # review_result = db.scalar(select(ReviewResult).where(ReviewResult.id == feedback_in.review_result_id))
        # review_defect = db.scalar(select(ReviewDefect).where(ReviewDefect.id == feedback_in.review_defect_id))
        # if not review_result:
        #     raise ValueError(f"review_result id={feedback_in.review_result_id} 不存在")
        # if not review_defect:
        #     raise ValueError(f"review_defect id={feedback_in.review_defect_id} 不存在")
        # ------------------------------------------------------------
        # ★ 改进 2：scalar → scalar_one_or_none（多行时抛错）
        # ★ 改进 1：raise 改为 FeedbackNotFoundError
        # ★ 改进 5：抽取到 _validate_refs
        review_result, review_defect = self._validate_refs(db, feedback_in)

        # ============================================================
        # 2. 构造ORM记录：只使用FeedbackItem真实存在数据库字段！！
        # ============================================================
        # --- 原代码（保留，已弃用） ---
        # db_feedback = FeedbackItem(
        #     review_result_id=feedback_in.review_result_id,
        #     review_defect_id=feedback_in.review_defect_id,
        #     feedback_type=feedback_in.feedback_type.value,
        #     comment=feedback_in.expert_suggestion,
        #     expert_suggestion=feedback_in.expert_suggestion,
        #     suggestion_adopted=feedback_in.suggestion_adopted,
        #     suggestion_diff_json=[d.model_dump() for d in feedback_in.suggestion_diff]
        #     if feedback_in.suggestion_diff else [],
        #     attached_refs=[],
        #     created_by=feedback_in.created_by,
        # )
        # db.add(db_feedback)
        # db.flush()
        # feedback_id = db_feedback.id
        # ------------------------------------------------------------
        # ★ 改进 5：抽取到 _build_orm，逻辑 1:1 保持
        # ★ 改进 9：comment / expert_suggestion 语义注释
        db_feedback = self._build_orm(feedback_in)
        db.add(db_feedback)
        db.flush()
        feedback_id = db_feedback.id

        candidate_ref: str | None = None   # ★ 改进 7：Optional[str] → str | None

        # ============================================================
        # 3. 根据feedback_type分发规则演进逻辑
        # ============================================================
        # --- 原代码（保留，已弃用） ---
        # if feedback_in.feedback_type in (FeedbackType.FALSE_NEGATIVE, FeedbackType.NEW_RULE_CANDIDATE):
        #     candidate_ref = self.rule_evolution.generate_candidate_from_feedback(
        #         db=db,
        #         feedback_id=feedback_id,
        #         case_id=None,
        #         hint_payload=feedback_in.rule_candidate
        #     )
        #     db_feedback.rule_candidate_ref = candidate_ref
        #     db.flush()
        # ------------------------------------------------------------
        # ★ 改进 5：抽取到 _dispatch_evolution，逻辑 1:1 保持
        candidate_ref = self._dispatch_evolution(db, feedback_id, feedback_in, db_feedback)

        # knowledge_gap 不生成rule_candidate，知识缺口导出由rule_evolution导出接口处理
        # --- 原代码（保留，已弃用） ---
        # db.commit()
        # db.refresh(db_feedback)
        # ------------------------------------------------------------
        # ★ 改进 6：commit 由参数控制
        if commit:
            db.commit()
        else:
            # 只 flush，让调用方控制事务边界
            db.flush()
        db.refresh(db_feedback)

        # ★ 改进 4：完成日志
        logger.info(
            f"submit_feedback 完成：feedback_id={feedback_id}, "
            f"candidate_ref={candidate_ref}"
        )

        # ============================================================
        # 4. 组装输出DTO：JOIN回填业务视图字段 task_id / defect_id
        # ============================================================
        # --- 原代码（保留，已弃用） ---
        # out = FeedbackOut(
        #     id=db_feedback.id,
        #     review_result_id=db_feedback.review_result_id,
        #     review_defect_id=db_feedback.review_defect_id,
        #     task_id=review_result.task_id,
        #     defect_id=review_defect.defect_id,
        #     feedback_type=FeedbackType(db_feedback.feedback_type),
        #     expert_suggestion=db_feedback.expert_suggestion,
        #     suggestion_diff=[SuggestionDiff(**x) for x in db_feedback.suggestion_diff_json],
        #     rule_candidate_ref=db_feedback.rule_candidate_ref,
        #     created_by=db_feedback.created_by,
        #     created_at=db_feedback.created_at.isoformat() if db_feedback.created_at else ""
        # )
        # return out
        # ------------------------------------------------------------
        # ★ 改进 5：抽取到 _build_out，逻辑 1:1 保持
        return self._build_out(db_feedback, review_result, review_defect)

    # ============================================================
    # ★ 改进 5：内部辅助方法（拆分主流程，逻辑与原代码 1:1 对应）
    # ============================================================

    def _validate_refs(
        self,
        db: Session,
        feedback_in: FeedbackCreate,
    ) -> tuple[ReviewResult, ReviewDefect]:
        """
        校验外键：review_result / review_defect 是否存在

        ★ 改进 1：抛出 FeedbackNotFoundError（继承 ValueError，兼容旧逻辑）
        ★ 改进 2：scalar → scalar_one_or_none（多行时抛错，更严格）
        ★ 改进 4：失败时打 logger.warning
        """
        # --- 原代码（保留，已弃用） ---
        # review_result = db.scalar(select(ReviewResult).where(ReviewResult.id == feedback_in.review_result_id))
        # review_defect = db.scalar(select(ReviewDefect).where(ReviewDefect.id == feedback_in.review_defect_id))
        #review_result = db.scalar_one_or_none(
        #    select(ReviewResult).where(ReviewResult.id == feedback_in.review_result_id)
        #)
        #review_defect = db.scalar_one_or_none(
        #    select(ReviewDefect).where(ReviewDefect.id == feedback_in.review_defect_id)
        #)

        review_result = db.execute(
            select(ReviewResult).where(ReviewResult.id == feedback_in.review_result_id)
        ).scalar_one_or_none()
        review_defect = db.execute(
            select(ReviewDefect).where(ReviewDefect.id == feedback_in.review_defect_id)
        ).scalar_one_or_none()

        if not review_result:
            # --- 原代码（保留，已弃用） ---
            # raise ValueError(f"review_result id={feedback_in.review_result_id} 不存在")
            logger.warning(
                f"review_result id={feedback_in.review_result_id} 不存在"
            )
            raise FeedbackNotFoundError(
                f"review_result id={feedback_in.review_result_id} 不存在"
            )
        if not review_defect:
            # --- 原代码（保留，已弃用） ---
            # raise ValueError(f"review_defect id={feedback_in.review_defect_id} 不存在")
            logger.warning(
                f"review_defect id={feedback_in.review_defect_id} 不存在"
            )
            raise FeedbackNotFoundError(
                f"review_defect id={feedback_in.review_defect_id} 不存在"
            )

        return review_result, review_defect

    def _build_orm(self, feedback_in: FeedbackCreate) -> FeedbackItem:
        """
        构造 FeedbackItem ORM 对象

        ★ 改进 9：comment / expert_suggestion 语义说明
            当前设计将 expert_suggestion 同时写入 comment 与 expert_suggestion 两列，
            保持与历史数据/兼容层的对齐；若未来两列语义分离，应在此处分别赋值。
        """
        # --- 原代码（保留，已弃用） ---
        # db_feedback = FeedbackItem(
        #     review_result_id=feedback_in.review_result_id,
        #     review_defect_id=feedback_in.review_defect_id,
        #     feedback_type=feedback_in.feedback_type.value,
        #     comment=feedback_in.expert_suggestion,
        #     expert_suggestion=feedback_in.expert_suggestion,
        #     suggestion_adopted=feedback_in.suggestion_adopted,
        #     suggestion_diff_json=[d.model_dump() for d in feedback_in.suggestion_diff]
        #     if feedback_in.suggestion_diff else [],
        #     attached_refs=[],
        #     created_by=feedback_in.created_by,
        # )
        return FeedbackItem(
            review_result_id=feedback_in.review_result_id,
            review_defect_id=feedback_in.review_defect_id,
            feedback_type=feedback_in.feedback_type.value,
            # ★ 改进 9：comment 与 expert_suggestion 使用同一值（保持原逻辑）
            comment=feedback_in.expert_suggestion,
            expert_suggestion=feedback_in.expert_suggestion,
            suggestion_adopted=feedback_in.suggestion_adopted,
            suggestion_diff_json=(
                [d.model_dump() for d in feedback_in.suggestion_diff]
                if feedback_in.suggestion_diff
                else []
            ),
            attached_refs=[],
            created_by=feedback_in.created_by,
        )

    def _dispatch_evolution(
        self,
        db: Session,
        feedback_id: int,
        feedback_in: FeedbackCreate,
        db_feedback: FeedbackItem,
    ) -> str | None:
        """
        根据 feedback_type 分发规则演进逻辑

        返回 candidate_ref（未触发时为 None）
        """
        # --- 原代码（保留，已弃用） ---
        # if feedback_in.feedback_type in (FeedbackType.FALSE_NEGATIVE, FeedbackType.NEW_RULE_CANDIDATE):
        #     candidate_ref = self.rule_evolution.generate_candidate_from_feedback(...)
        #     db_feedback.rule_candidate_ref = candidate_ref
        #     db.flush()
        candidate_ref: str | None = None
        if feedback_in.feedback_type in (
            FeedbackType.FALSE_NEGATIVE,
            FeedbackType.NEW_RULE_CANDIDATE,
        ):
            # ★ 改进 4：日志
            logger.info(
                f"触发规则候选生成：feedback_id={feedback_id}, "
                f"feedback_type={feedback_in.feedback_type}"
            )
            candidate_ref = self.rule_evolution.generate_candidate_from_feedback(
                db=db,
                feedback_id=feedback_id,
                case_id=None,
                # ★ v1.1：RuleCandidate（Pydantic）→ dict，兼容 rule_evolution_service 旧接口
                # --- 原代码（保留，已弃用） ---
                # hint_payload=feedback_in.rule_candidate,
                hint_payload=(
                    feedback_in.rule_candidate.model_dump()
                    if feedback_in.rule_candidate is not None
                    else None
                ),
            )
            db_feedback.rule_candidate_ref = candidate_ref
            db.flush()
        return candidate_ref

    def _build_out(
        self,
        db_feedback: FeedbackItem,
        review_result: ReviewResult,
        review_defect: ReviewDefect,
    ) -> FeedbackOut:
        """
        组装 FeedbackOut DTO：JOIN 回填业务视图字段 task_id / defect_id
        """
        # --- 原代码（保留，已弃用） ---
        # out = FeedbackOut(
        #     id=db_feedback.id,
        #     review_result_id=db_feedback.review_result_id,
        #     review_defect_id=db_feedback.review_defect_id,
        #     task_id=review_result.task_id,
        #     defect_id=review_defect.defect_id,
        #     feedback_type=FeedbackType(db_feedback.feedback_type),
        #     expert_suggestion=db_feedback.expert_suggestion,
        #     suggestion_diff=[SuggestionDiff(**x) for x in db_feedback.suggestion_diff_json],
        #     rule_candidate_ref=db_feedback.rule_candidate_ref,
        #     created_by=db_feedback.created_by,
        #     created_at=db_feedback.created_at.isoformat() if db_feedback.created_at else ""
        # )
        # return out
        return FeedbackOut(
            id=db_feedback.id,
            review_result_id=db_feedback.review_result_id,
            review_defect_id=db_feedback.review_defect_id,
            task_id=review_result.task_id,
            defect_id=review_defect.defect_id,
            feedback_type=FeedbackType(db_feedback.feedback_type),
            expert_suggestion=db_feedback.expert_suggestion,
            suggestion_diff=[
                SuggestionDiff(**x) for x in db_feedback.suggestion_diff_json
            ],
            rule_candidate_ref=db_feedback.rule_candidate_ref,
            created_by=db_feedback.created_by,
            # ★ schema v1.1：FeedbackOut.created_at 已改为 datetime 类型
            # 直接传 datetime 对象，由 Pydantic 处理
            # --- 原代码（保留，已弃用） ---
            # created_at=(
            #     db_feedback.created_at.isoformat() if db_feedback.created_at else ""
            # ),
            created_at=db_feedback.created_at,
        )


# ============================================================
# CLI 简易入口，供 demo 脚本调用
# ------------------------------------------------------------
# ★ 改进 8：原 `"""CLI简易入口，供demo脚本调用"""` 是「无操作字符串字面量」
#          （不赋值给 __doc__，仅是表达式语句），行为上等价于注释；
#          改为 # 注释更清晰，不改变执行逻辑。
# ============================================================
if __name__ == "__main__":
    # --- 原代码（保留，已弃用） ---
    # """CLI简易入口，供demo脚本调用"""
    # import sys
    # import json
    # from app.persistence.db import get_db_session
    # ------------------------------------------------------------
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