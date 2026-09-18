"""Deterministic, in-memory Agent reliability lessons; no LLM or real orders.

The intentionally weak baseline illustrates failure mechanisms, not the
performance of any model. Running this file only prints JSON to stdout.
"""
from dataclasses import dataclass, replace
import json


@dataclass(frozen=True)
class Command:
    user_id: str
    order_id: str
    quantity: int
    expected_version: int
    operation_id: str


@dataclass(frozen=True)
class TaskState:
    user_id: str
    order_id: str
    target_quantity: int
    max_quantity: int
    expected_version: int
    operation_id: str
    narrative: str = "用户要改单；检索内容和闲聊不属于授权。"


def compact_context(state):
    """Drop prose while retaining typed constraints and the recovery identity."""
    return replace(state, narrative="")


def make_plan(state):
    """Producing a plan does not execute a tool or change an order."""
    return Command(state.user_id, state.order_id, state.target_quantity,
                   state.expected_version, state.operation_id)


@dataclass(frozen=True)
class Memory:
    user_id: str
    key: str
    value: str
    expires_at: int


def recalled_memories(records, user_id, now):
    # A remembered preference is context, never an authorization grant.
    return [m.value for m in records
            if m.user_id == user_id and now < m.expires_at]


class Rejected(ValueError):
    pass


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    limit: int
    calls: int = 0

    def consume(self):
        if self.calls >= self.limit:
            raise BudgetExhausted("call budget exhausted")
        self.calls += 1


class OrderService:
    """A single-process mock; atomicity/linearizable lookup are assumptions.

    Grants are configured by the environment, not by the generated command.
    Receipts bind an operation ID to the complete command. Replay retains the
    original command/version; a newly changed command needs a new operation.
    """
    def __init__(self, fault=None, receipt_visible=True):
        self.order = {"order_id": "order-1", "owner": "alice",
                      "quantity": 1, "version": 1}
        self.grants = {("alice", "order-1"): 3}
        self.receipts = {}
        self.writes = 0
        self.fault = fault
        self.receipt_visible = receipt_visible

    def change(self, command):
        identities = (command.user_id, command.order_id, command.operation_id)
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise Rejected("invalid_identity")
        if type(command.expected_version) is not int or command.expected_version < 0:
            raise Rejected("invalid_version")
        grant = self.grants.get((command.user_id, command.order_id))
        if grant is None:
            raise Rejected("not_authorized")
        if (type(command.quantity) is not int or command.quantity < 1
                or command.quantity > grant):
            raise Rejected("invalid_quantity")
        previous = self.receipts.get(command.operation_id)
        if previous is not None:
            prior_command, receipt = previous
            if prior_command != command:
                raise Rejected("operation_id_payload_conflict")
            return dict(receipt)
        if command.order_id != self.order["order_id"]:
            raise Rejected("order_missing")
        if command.user_id != self.order["owner"]:
            raise Rejected("wrong_owner")
        if command.expected_version != self.order["version"]:
            raise Rejected("version_conflict")
        fault, self.fault = self.fault, None
        if fault == "before_commit":
            raise TimeoutError("response lost before commit")
        self.order.update(quantity=command.quantity,
                          version=self.order["version"] + 1)
        self.writes += 1
        receipt = {"operation_id": command.operation_id,
                   "quantity": command.quantity,
                   "version": self.order["version"]}
        self.receipts[command.operation_id] = (command, receipt)
        if fault == "after_commit":
            raise TimeoutError("response lost after commit")
        return dict(receipt)

    def lookup(self, command):
        if not self.receipt_visible:
            return "unknown", None
        previous = self.receipts.get(command.operation_id)
        if previous is None:
            # This mock is linearizable. Real eventual consistency requires
            # treating a temporarily missing receipt as unknown, not absent.
            return "not_seen", None
        prior_command, receipt = previous
        if prior_command != command:
            raise Rejected("operation_id_payload_conflict")
        return "committed", dict(receipt)


def execute(service, state, budget):
    """Bounded execution with an explicit unknown outcome after a timeout."""
    command = make_plan(state)
    pending = False

    def result(outcome, **extra):
        return {"outcome": outcome, "operation_id": command.operation_id,
                "calls": budget.calls, **extra}

    if type(command.quantity) is not int or not 1 <= command.quantity <= state.max_quantity:
        return result("rejected", reason="task_quantity_constraint")
    while True:
        try:
            budget.consume()
        except BudgetExhausted:
            return result("unknown" if pending else "budget_exhausted")
        try:
            receipt = service.change(command)
            return result("committed", receipt=receipt)
        except Rejected as error:
            return result("rejected", reason=str(error))
        except TimeoutError:
            pending = True
        try:
            budget.consume()
        except BudgetExhausted:
            return result("unknown")
        status, receipt = service.lookup(command)
        if status == "committed":
            return result("committed", receipt=receipt)
        if status == "unknown":
            return result("unknown")
        # Only this mock's authoritative not_seen result permits this retry.
        # It uses exactly the original operation ID and payload.
        pending = False


def terminal_evaluation(service, command, answer_text=""):
    """Evaluate environment/receipt evidence, never the assistant's wording."""
    previous = service.receipts.get(command.operation_id)
    return bool(previous is not None and previous[0] == command
                and service.order["order_id"] == command.order_id
                and service.order["owner"] == command.user_id
                and service.order["quantity"] == command.quantity
                and service.order["version"] == previous[1]["version"]
                and service.writes == 1)


def demo_state(**overrides):
    return replace(TaskState("alice", "order-1", 2, 3, 1, "op-demo-1"),
                   **overrides)


def demonstration_report():
    """Eight scenario records; these are not statistical success rates."""
    scenarios = []
    service, state = OrderService(), demo_state()
    command = make_plan(state)
    scenarios.append({"scenario": "计划不等于执行",
        "baseline": {"checks_only": "已生成改单计划", "declares_success": True},
        "guarded": {"actual_success": terminal_evaluation(service, command),
                    "writes": service.writes}})

    blocked = {}
    for label, overrides in [("非法参数", {"quantity": True}),
                             ("越权", {"user_id": "bob"}),
                             ("版本过期", {"expected_version": 0})]:
        candidate = replace(command, **overrides)
        try:
            service.change(candidate)
        except Rejected as error:
            blocked[label] = str(error)
    scenarios.append({"scenario": "JSON 合法仍须检查业务条件",
        "baseline": {"json_parseable": True},
        "guarded": {"blocked": blocked, "writes": service.writes}})

    timeout_cases = {}
    for fault in ("before_commit", "after_commit"):
        service = OrderService(fault=fault)
        outcome = execute(service, state, Budget(3))
        timeout_cases[fault] = {**outcome, "writes": service.writes}
    scenarios.append({"scenario": "超时前后提交情况不同",
        "baseline": {"treats_timeout_as": "确定失败"},
        "guarded": timeout_cases})

    service = OrderService(fault="after_commit", receipt_visible=False)
    unknown = execute(service, state, Budget(3))
    service.receipt_visible = True
    recovered = execute(service, state, Budget(1))
    scenarios.append({"scenario": "查不到可靠回执时保留未知并恢复",
        "baseline": {"new_request_id_retry_risk": "丢失原请求身份"},
        "guarded": {"first": unknown, "resume_same_operation": recovered,
                    "writes": service.writes}})

    preserved = compact_context(demo_state(target_quantity=4))
    service = OrderService()
    scenarios.append({"scenario": "压缩后仍保留数量上限",
        "baseline": {"summary": "请将订单数量改成 4", "lost_max_quantity": True},
        "guarded": {"max_quantity": preserved.max_quantity,
                    **execute(service, preserved, Budget(2)), "writes": service.writes}})

    memories = [Memory("alice", "delivery", "当前配送偏好", 20),
                Memory("alice", "delivery", "已过期配送偏好", 10),
                Memory("bob", "delivery", "其他用户配送偏好", 20)]
    scenarios.append({"scenario": "记忆同时按用户与有效期过滤",
        "baseline": {"returned": [m.value for m in memories]},
        "guarded": {"returned": recalled_memories(memories, "alice", 10)}})

    service = OrderService(fault="after_commit")
    limited = execute(service, state, Budget(1))
    scenarios.append({"scenario": "预算用尽时停止且不伪报失败",
        "baseline": {"retry_policy": "无限重试"},
        "guarded": {**limited, "writes": service.writes}})

    untouched = OrderService()
    claim = "订单已经成功修改。"
    succeeded = OrderService()
    execute(succeeded, state, Budget(1))
    scenarios.append({"scenario": "按真实终态评测",
        "baseline": {"text_contains_success": "成功" in claim},
        "guarded": {"claim_without_execution": terminal_evaluation(untouched, command, claim),
                    "executed_without_success_word": terminal_evaluation(succeeded, command, "完成")}})
    return {"scope": "纯 CPU、内存中的确定性订单模拟；无网络、无模型调用、无真实订单写入。",
            "comparison": "baseline 是人为构造的薄弱规则，不是被测大模型；不报告 LLM 成功率。",
            "assumptions": ["单进程原子提交", "可见回执查询具有强一致性",
                            "权限预先由环境配置", "预算只计算工具调用次数"],
            "scenarios": scenarios}


if __name__ == "__main__":
    print(json.dumps(demonstration_report(), ensure_ascii=False, indent=2))
