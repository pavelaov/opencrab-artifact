"""Tests for opencrab2.highlevel.contract."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import pytest

from opencrab2.errors import ContractViolation
from opencrab2.highlevel.contract import (CONTRACTS, DEFAULT_CONTRACT, SMOLLM_CONTRACT, NumericContract,
                                          canonical_json_bytes, contract_by_name)

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent / "opencrab2"


def test_default_fields_match_spec_table():
    c = DEFAULT_CONTRACT
    assert c.ring_bits == 64 and c.frac_bits == 16 and c.logical_bits == 40 and c.dcf_domain_bits == 32
    assert c.name == "NC-v2-f16-l40"
    assert c.gelu == "gelu_clip12" and c.silu == "silu_clip12"
    assert (c.exp_order, c.exp_r, c.exp_Z, c.exp_extra_bits) == (2, 7, 16, 4)
    assert (c.recip_K, c.recip_degree, c.recip_newton) == (11, 5, 2)
    assert (c.rsqrt_j0, c.rsqrt_m, c.rsqrt_degree, c.rsqrt_newton) == (-7, 15, 5, 2)
    assert c.layernorm_extra_bits == 4
    assert c.argmax_tie == "lowest_index"
    assert c.trunc_products == "local_prob" and c.trunc_square_ln == "faithful"
    assert c.activation_semantics == "erf_gelu"
    assert c.scale == 65536 and c.logical_bound == 2 ** 39
    assert c.compare_bound_real == 2.0 ** 15


def test_hash_is_sha256_of_canonical_json():
    c = DEFAULT_CONTRACT
    expected = hashlib.sha256(
        json.dumps(dataclasses.asdict(c), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert c.contract_hash() == expected
    assert len(c.contract_hash()) == 64
    assert c.contract_hash() == NumericContract().contract_hash()
    assert canonical_json_bytes({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(NumericContract)])
def test_every_field_changes_the_hash(field):
    base = DEFAULT_CONTRACT
    cur = getattr(base, field)
    if isinstance(cur, bool):
        new = not cur
    elif isinstance(cur, int):
        new = cur + 1
        if field == "ring_bits":
            pytest.skip("ring_bits is fixed to 64 by the contract validation")
        if field == "exp_order":
            new = 1
        if field == "frac_bits":
            new = 15
        if field == "exp_r":
            new = 6
    else:
        alternatives = {"name": cur + "x", "gelu": "gelu_clip8", "silu": "silu_clip8",
                        "trunc_products": "faithful", "trunc_square_ln": "local_prob"}
        new = alternatives.get(field)
        if new is None:
            pytest.skip(f"{field} admits a single value")
    other = dataclasses.replace(base, **{field: new})
    assert other.contract_hash() != base.contract_hash()


def test_smollm_contract_differs_only_in_documented_fields():
    a = DEFAULT_CONTRACT.to_json()
    b = SMOLLM_CONTRACT.to_json()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"logical_bits", "rsqrt_m", "name"}
    assert SMOLLM_CONTRACT.logical_bits == 46 and SMOLLM_CONTRACT.rsqrt_m == 16
    assert SMOLLM_CONTRACT.name == "NC-v2-f16-l46-smollm"
    assert SMOLLM_CONTRACT.contract_hash() != DEFAULT_CONTRACT.contract_hash()


def test_json_round_trip_and_strictness():
    d = SMOLLM_CONTRACT.to_json()
    assert NumericContract.from_json(d) == SMOLLM_CONTRACT
    bad = dict(d)
    bad["extra"] = 1
    with pytest.raises(ContractViolation) as ei:
        NumericContract.from_json(bad)
    assert str(ei.value).startswith("CONTRACT_JSON:")
    missing = dict(d)
    del missing["frac_bits"]
    with pytest.raises(ContractViolation):
        NumericContract.from_json(missing)


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONTRACT.frac_bits = 12   # type: ignore[misc]


@pytest.mark.parametrize("kw", [
    {"ring_bits": 32}, {"frac_bits": 0}, {"frac_bits": 40, "logical_bits": 40}, {"logical_bits": 64},
    {"dcf_domain_bits": 65}, {"dcf_domain_bits": 1}, {"exp_order": 3}, {"exp_r": 0}, {"exp_r": 17},
    {"argmax_tie": "random"}, {"trunc_products": "none"}, {"trunc_square_ln": "magic"},
    {"activation_semantics": "gelu_new"}, {"name": ""}, {"recip_K": -1},
])
def test_invalid_parameters_fail_closed(kw):
    with pytest.raises(ContractViolation) as ei:
        NumericContract(**kw)
    assert str(ei.value).startswith("CONTRACT_PARAMS:")
    assert ei.value.code == "CONTRACT_PARAMS"


def test_softmax_extra_bits():
    c = DEFAULT_CONTRACT
    assert c.softmax_extra_bits(1) == 0
    assert c.softmax_extra_bits(2) == 1
    assert c.softmax_extra_bits(16) == 4
    assert c.softmax_extra_bits(17) == 5
    assert c.softmax_extra_bits(4096) == 10
    assert c.softmax_extra_bits(100000) == 10
    with pytest.raises(ContractViolation):
        c.softmax_extra_bits(0)


def test_contract_by_name():
    assert contract_by_name("NC-v2-f16-l40") is DEFAULT_CONTRACT
    assert contract_by_name("NC-v2-f16-l46-smollm") is SMOLLM_CONTRACT
    assert set(CONTRACTS) == {"NC-v2-f16-l40", "NC-v2-f16-l46-smollm"}
    with pytest.raises(ContractViolation) as ei:
        contract_by_name("NC-v9")
    assert ei.value.code == "CONTRACT_UNKNOWN"


def test_replace_changes_hash_and_keeps_validation():
    c = DEFAULT_CONTRACT.replace(frac_bits=12)
    assert c.frac_bits == 12 and c.contract_hash() != DEFAULT_CONTRACT.contract_hash()
    with pytest.raises(ContractViolation):
        DEFAULT_CONTRACT.replace(logical_bits=10)


@pytest.mark.parametrize("rel", [
    "highlevel/contract.py", "highlevel/channel.py", "highlevel/ledger.py", "highlevel/dealer.py",
    "highlevel/telemetry.py", "platform/dealer_service.py",
])
def test_no_em_or_en_dashes_in_sources(rel):
    """Coding rule: no em dashes (U+2014) or en dashes (U+2013) anywhere."""
    text = (PKG / rel).read_text(encoding="utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text


@pytest.mark.parametrize("name", [
    "test_contract.py", "test_channel.py", "test_ledger.py", "test_dealer.py", "test_telemetry.py",
    "test_dealer_service.py",
])
def test_no_em_or_en_dashes_in_tests(name):
    text = (HERE / name).read_text(encoding="utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text
