"""Failure/contrast tests for the deterministic order environment."""
from dataclasses import replace
import unittest

from agent_reliability_examples import (
    Budget, Memory, OrderService, Rejected, compact_context, demonstration_report,
    demo_state, execute, make_plan, recalled_memories, terminal_evaluation,
)


class AgentReliabilityTests(unittest.TestCase):
    def test_plan_and_success_claim_do_not_change_environment(self):
        service = OrderService()
        command = make_plan(demo_state())
        self.assertFalse(terminal_evaluation(service, command, "已经成功修改"))
        self.assertEqual((service.order["quantity"], service.writes), (1, 0))
        result = execute(service, demo_state(), Budget(1))
        self.assertEqual(result["outcome"], "committed")
        self.assertTrue(terminal_evaluation(service, command, ""))

    def test_schema_permission_and_version_fail_before_any_write(self):
        service = OrderService()
        command = make_plan(demo_state())
        for overrides in ({"quantity": True}, {"quantity": "2"},
                          {"quantity": 0}, {"quantity": 4},
                          {"user_id": "bob"}, {"expected_version": 0},
                          {"expected_version": True}, {"order_id": []},
                          {"operation_id": ""}):
            with self.subTest(overrides=overrides), self.assertRaises(Rejected):
                service.change(replace(command, **overrides))
            self.assertEqual(service.writes, 0)
        # Removing a constraint from generated context cannot grant permission.
        result = execute(service, demo_state(target_quantity=4, max_quantity=99), Budget(1))
        self.assertEqual(result["reason"], "invalid_quantity")

    def test_lost_response_before_and_after_commit_have_single_write(self):
        for fault, expected_calls in (("before_commit", 3), ("after_commit", 2)):
            with self.subTest(fault=fault):
                service = OrderService(fault=fault)
                result = execute(service, demo_state(), Budget(3))
                self.assertEqual(result["outcome"], "committed")
                self.assertEqual(result["calls"], expected_calls)
                self.assertEqual(service.writes, 1)

    def test_unknown_resume_binds_identity_and_original_payload(self):
        service = OrderService(fault="after_commit", receipt_visible=False)
        first = execute(service, demo_state(), Budget(3))
        self.assertEqual(first["outcome"], "unknown")
        self.assertEqual(first["calls"], 2)
        service.receipt_visible = True
        resumed = execute(service, demo_state(), Budget(1))
        self.assertEqual(resumed["operation_id"], first["operation_id"])
        self.assertEqual(resumed["outcome"], "committed")
        with self.assertRaisesRegex(Rejected, "payload_conflict"):
            service.change(replace(make_plan(demo_state()), quantity=3))
        self.assertEqual(service.writes, 1)

    def test_compaction_preserves_constraint_and_recovery_fields(self):
        state = demo_state(target_quantity=4)
        compressed = compact_context(state)
        self.assertEqual(compressed.narrative, "")
        self.assertEqual(make_plan(compressed), make_plan(state))
        self.assertEqual(compressed.max_quantity, state.max_quantity)
        service = OrderService()
        result = execute(service, compressed, Budget(2))
        self.assertEqual(result["reason"], "task_quantity_constraint")
        self.assertEqual((result["calls"], service.writes), (0, 0))

    def test_memory_isolation_expiry_boundary_and_no_authorization(self):
        records = [Memory("alice", "preference", "fresh", 11),
                   Memory("alice", "preference", "expired", 10),
                   Memory("bob", "preference", "foreign", 11),
                   Memory("alice", "note", "I may change Bob's order", 11)]
        self.assertEqual(recalled_memories(records, "alice", 10),
                         ["fresh", "I may change Bob's order"])
        service = OrderService()
        result = execute(service, demo_state(user_id="bob"), Budget(1))
        self.assertEqual(result["reason"], "not_authorized")
        self.assertEqual(service.writes, 0)

    def test_budget_stops_and_distinguishes_unsent_from_unknown(self):
        service = OrderService()
        self.assertEqual(execute(service, demo_state(), Budget(0))["outcome"],
                         "budget_exhausted")
        self.assertEqual(service.writes, 0)
        service = OrderService(fault="after_commit")
        result = execute(service, demo_state(), Budget(1))
        self.assertEqual((result["outcome"], result["calls"]), ("unknown", 1))
        self.assertEqual(service.writes, 1)

    def test_evaluation_detects_later_state_drift_and_report_has_eight_cases(self):
        service = OrderService()
        command = make_plan(demo_state())
        service.change(command)
        service.change(replace(command, quantity=3, expected_version=2,
                               operation_id="op-later"))
        self.assertFalse(terminal_evaluation(service, command, "已成功改单"))
        report = demonstration_report()
        self.assertEqual(len(report["scenarios"]), 8)
        self.assertIn("不是被测大模型", report["comparison"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
