"""Print numerical teaching examples; never contacts a model API."""
import json
from pathlib import Path
from core_algorithms import *

out = {
    "description": "CPU教学算例，不是大模型训练实验",
    "dpo_initial_loss": float(dpo_loss(-2, -3, -2, -3)),
    "dpo_P09_example": float(dpo_loss(-100, -30, -110, -32, .1)),
    "simpo_P09_example": float(simpo_loss(-100, -30, 100, 20)),
    "grpo_advantages": grpo_advantages([[0, 0, 1, 1], [1, 1, 1, 1]]).tolist(),
    "kto_at_reference": kto_loss([0, 0], [True, False]).tolist(),
    "pass_at_4_if_iid_p_0_8": 1-(1-.8)**4,
    "pass_all_4_if_iid_p_0_8": .8**4,
    "kv_cache_gib": kv_cache_bytes(32, 1, 8192, 8, 128)/2**30,
    "precision_recall_f1": classification_metrics(80, 20, 40),
    "paired_bootstrap_identical": paired_bootstrap([1,0,1,1], [1,0,1,1]),
    "rrf": reciprocal_rank_fusion([["A","B","C"], ["C","B","A"]])
}
service = MockRefundService()
out["refund_final_state"] = recover_refund(service, "order-demo", "request-demo")
out["refund_write_count"] = service.writes
text = json.dumps(out, ensure_ascii=False, indent=2)
(Path(__file__).parent / "教学运行结果.json").write_text(text, encoding="utf-8")
print(text)
