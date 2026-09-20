"""Inspectable, bounded evidence whitelist. Never scan a user's directory."""
import math
import re

HASH = re.compile(r'^[a-f0-9]{64}$')
METRICS = ('accuracy', 'rmse', 'recall_at_5', 'ndcg_at_10', 'cluster_agreement',
           'evaluated_rows', 'all_rows', 'queries', 'nonfinite_rows')
FLAGS = ('allow_tf32', 'allow_fp16_reduced_precision_reduction', 'allow_bf16_reduced_precision_reduction')


def valid_hash(value):
    return value if isinstance(value, str) and HASH.fullmatch(value) else None


def scenario_evidence(card):
    from .scenarios.registry import ORDER, config_id
    rows = []
    for row in card.get('scenarios', [])[:24]:
        if row.get('scenario') not in ORDER or not row.get('config'):
            continue
        metrics = {k: float(v) for k, v in (row.get('metric') or {}).items()
                   if k in METRICS and type(v) in (int, float) and math.isfinite(v)}
        flags = {k: v if type(v) is bool else None for k, v in (row.get('math_flags') or {}).items() if k in FLAGS}
        decision = row.get('decision_audit') or {}
        rows.append({'config_id': config_id(row['config']),
                     'precision': row.get('precision') if row.get('precision') in ('bf16', 'fp16', 'fp32') else None,
                     'arithmetic': 'NOT_ASSESSED', 'metrics': metrics,
                     'scores_sha256': valid_hash(row.get('scores_sha256')),
                     'dataset_sha256': valid_hash(row.get('dataset_sha256')),
                     'model_sha256': valid_hash(row.get('model_sha256')),
                     'math_flags': flags,
                     'decision_order_sha256': valid_hash(decision.get('order_sha256')),
                     'decision_membership_sha256': valid_hash(decision.get('membership_sha256')),
                     'decision_rule': decision.get('rule') if decision.get('rule') in ('stable_descending_index', 'argmax_first_index') else None,
                     'nonfinite_count': int(decision.get('nonfinite_count', 0)),
                     'comparator': 'NONE_NO_HARDWARE_VERDICT'})
    return rows
