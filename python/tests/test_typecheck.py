"""Tests for opencrab2.highlevel.typecheck: domain inference and the typing reason codes."""

from __future__ import annotations

from pathlib import Path

import pytest

from opencrab2.errors import AdmissionError
from opencrab2.highlevel import registry as rg
from opencrab2.highlevel import typecheck as tc

PLAN = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}


def head(kind_emb="embedding_wmask"):
    return [{"kind": "input_client_plain", "out": "tok", "args": {}},
            {"kind": "share_input", "out": "onehot", "args": {"x": "tok"}},
            {"kind": kind_emb, "out": "x0", "args": {"x": "onehot"}}]


def code_of(tape):
    with pytest.raises(AdmissionError) as info:
        tc.check_tape(tape)
    assert str(info.value).startswith(info.value.code + ":")
    return info.value.code


def test_default_tape_types_end_to_end():
    tape = rg.default_gpt2_tape(PLAN, n_layers=2)
    doms = tc.infer_domains(tape)
    assert doms["tok"] == "plain_client"
    assert doms["onehot"] == "A64" and doms["x0"] == "A64"
    assert doms["logits"] == "A64" and doms["idx"] == "A64"
    assert doms["token"] == "plain_client"
    assert all(d in tc.DOMAINS for d in doms.values())
    assert tc.tape_feasible(tape) == (True, "OK")
    tc.check_tape(tape)


def test_missing_conversions():
    # plaintext one-hot fed straight into a shared gadget: share_input is the missing conversion
    tape = [{"kind": "input_client_plain", "out": "tok", "args": {}},
            {"kind": "embedding_wmask", "out": "x0", "args": {"x": "tok"}}]
    assert code_of(tape) == "MISSING_CONVERSION"
    assert tc.tape_feasible(tape) == (False, "MISSING_CONVERSION")
    # a2h_exact accepts A64 or Zt; an H_BFV consumer fed A64 names a2h_exact as the repair
    assert tc.CONVERSIONS[("A64", "Zt")] == "ring_to_field"
    assert tc.CONVERSIONS[("A64", "H_BFV")] == "a2h_exact"
    with_r2f = head() + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}},
                         {"kind": "a2h_exact", "out": "ct", "args": {"x": "z"}}]
    doms = tc.infer_domains(with_r2f)
    assert doms["z"] == "Zt" and doms["ct"] == "H_BFV"
    direct = head() + [{"kind": "a2h_exact", "out": "ct", "args": {"x": "x0"}}]
    assert tc.infer_domains(direct)["ct"] == "H_BFV"


def test_domain_mismatches():
    # a Zt value cannot enter a Z_2^64 gadget (no registered conversion back)
    zt_into_ring = head() + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}},
                             {"kind": "layernorm_rows", "out": "y", "args": {"x": "z"}}]
    assert code_of(zt_into_ring) == "DOMAIN_MISMATCH"
    # an H_BFV ciphertext fed to the CKKS consumer (distinct types, nothing casts)
    bfv_into_ckks = head() + [{"kind": "a2h_exact", "out": "ct", "args": {"x": "x0"}},
                              {"kind": "ckks_fast_h2a", "out": "y", "args": {"x": "ct"}}]
    assert code_of(bfv_into_ckks) == "DOMAIN_MISMATCH"
    # a released plaintext fed back into a secure gadget: share_input is the registered repair
    fed_back = rg.default_gpt2_tape(PLAN, 1) + [{"kind": "gelu_clip12", "out": "g", "args": {"x": "token"}}]
    assert code_of(fed_back) == "MISSING_CONVERSION"
    # a server plaintext into a release (plain_server is not A64 and no conversion to plain_client exists)
    server_release = [{"kind": "input_server_plain", "out": "w", "args": {}},
                      {"kind": "release_final", "out": "t", "args": {"x": "w"}}]
    assert code_of(server_release) == "MISSING_CONVERSION"
    # undefined operand and missing operand
    assert code_of(head()[:2] + [{"kind": "relu", "out": "r", "args": {"x": "nowhere"}}]) == "DOMAIN_MISMATCH"
    assert code_of(head()[:2] + [{"kind": "relu", "out": "r", "args": {}}]) == "DOMAIN_MISMATCH"
    # a plaintext released to the client can be re-shared (next token one-hot)
    reshared = rg.default_gpt2_tape(PLAN, 1) + [{"kind": "share_input", "out": "next", "args": {"x": "token"}}]
    assert tc.infer_domains(reshared)["next"] == "A64"


def test_two_operand_ops_check_both():
    tape = head() + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}},
                     {"kind": "matmul_shared", "out": "m", "args": {"x": "x0", "y": "z"}}]
    assert code_of(tape) == "DOMAIN_MISMATCH"
    ok = head() + [{"kind": "matmul_shared", "out": "m", "args": {"x": "x0", "y": "x0"}}]
    assert tc.infer_domains(ok)["m"] == "A64"
    listed = head() + [{"kind": "matmul_shared", "out": "m", "args": {"inputs": ["x0", "x0"]}}]
    assert tc.infer_domains(listed)["m"] == "A64"


def test_unknown_kind_and_malformed_entries():
    assert code_of(head() + [{"kind": "he_linear", "out": "y", "args": {"x": "x0"}}]) == "NOT_ADMITTED_COMPONENT"
    assert code_of([{"kind": "input_client_plain"}]) == "DOMAIN_MISMATCH"
    assert code_of(["not a mapping"]) == "DOMAIN_MISMATCH"
    with pytest.raises(AdmissionError, match="^NOT_ADMITTED_COMPONENT"):
        tc.op_signature("ckks_fast_h2a_v2")
    assert tc.op_signature("input_server_plain") == (None, "plain_server")
    assert tc.op_signature("a2h_exact") == (("A64", "Zt"), "H_BFV")
    assert tc.split_domains("A64 | Zt") == ("A64", "Zt")


def test_disabled_components_type_but_are_not_admitted():
    # typing accepts the CKKS lane on an H_CKKS value; admission (registry) rejects it by status
    tape = [{"kind": "input_client_plain", "out": "tok", "args": {}}]
    doms = tc.infer_domains(tape)
    assert doms == {"tok": "plain_client"}
    ckks_tape = head() + [{"kind": "ckks_fast_h2a", "out": "y", "args": {"x": "x0"}}]
    assert code_of(ckks_tape) == "DOMAIN_MISMATCH"              # A64 is not H_CKKS and nothing produces H_CKKS
    assert rg.admit(ckks_tape, rg.PROFILE_ID, None) == (False, "NOT_ADMITTED_COMPONENT")


def test_no_em_dashes_in_source():
    src = Path(tc.__file__).read_text(encoding="utf-8")
    assert chr(0x2014) not in src and chr(0x2013) not in src
