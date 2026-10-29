from __future__ import annotations

import math


def resource_estimate(hypothesis_count: int, action_count: int = 7) -> dict[str, int]:
    hypothesis_bits = math.ceil(math.log2(hypothesis_count))
    binary_action_bits = math.ceil(math.log2(action_count))
    one_hot_qubits = hypothesis_bits + action_count
    amplitudes = 1 << one_hot_qubits
    return {
        "hypotheses": hypothesis_count,
        "hypothesis_bits": hypothesis_bits,
        "binary_action_bits": binary_action_bits,
        "one_hot_qaoa_qubits": one_hot_qubits,
        "statevector_amplitudes": amplitudes,
        "statevector_bytes_complex128": amplitudes * 16,
    }


def default_scaling_table() -> list[dict[str, int]]:
    return [resource_estimate(size) for size in (64, 256, 7_776, 262_144, 2_985_984, 16_777_216)]

