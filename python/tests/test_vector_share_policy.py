"""The dealer of the profile never holds a plaintext vector (offline means input independent; paper, Section 3.2).

``vector_share`` hands a server-private vector to the dealer in the clear. An eligible dealer refuses to
build such a bundle and a party refuses to use a dealer-provided share while its material client is
eligible; the type remains for the insecure unit-test harness.
"""
from __future__ import annotations

import numpy as np
import pytest

from opencrab2.errors import MaterialError, MaterialStall
from opencrab2.highlevel.contract import DEFAULT_CONTRACT
from opencrab2.highlevel.dealer import MODE_INSECURE, MODE_ISOLATED, Dealer
from opencrab2.highlevel.party import Party
from opencrab2.lowlevel.csprng import SecureRandom

C = DEFAULT_CONTRACT
V = np.arange(3, dtype=np.uint64)


def test_eligible_dealer_refuses_a_plaintext_vector():
    d = Dealer(C, None, MODE_ISOLATED)
    assert d.insecure is False
    for call in (lambda: d.make_vector_share("ln.b", V), lambda: d.make("vector_share", {"name": "ln.b", "values": [1, 2, 3]})):
        with pytest.raises(MaterialError) as ei:
            call()
        assert ei.value.code == "DEALER_PLAINTEXT_REFUSED"
    assert d.produced_summary() == {}                       # nothing was generated
    # every other type is still served, and the wmask path never takes W (shape only)
    assert d.make_beaver((4,), 1)[0][0]["insecure"] is False
    assert d.make_wmask("M", (2, 3))[0][0]["insecure"] is False


@pytest.mark.parametrize("dealer", [
    lambda: Dealer(C, np.random.default_rng(1), MODE_ISOLATED),          # seeded: insecure whatever the mode
    lambda: Dealer(C, SecureRandom.from_seed(1), MODE_ISOLATED),
    lambda: Dealer(C, None, MODE_INSECURE),
])
def test_insecure_harness_dealers_still_serve_the_type(dealer):
    d = dealer()
    (h0, p0, h1, p1), = d.make_vector_share("ln.b", V)
    assert h0["insecure"] is True and h1["insecure"] is True and h0["material_type"] == "vector_share"
    s0 = np.frombuffer(p0, dtype="<u8")
    s1 = np.frombuffer(p1, dtype="<u8")
    assert ((s0 + s1) & np.uint64(0xFFFFFFFFFFFFFFFF)).tolist() == V.tolist()


class _Front:
    """A material front end that returns a vector share and states its eligibility."""
    ledger = None

    def __init__(self, eligible, share=V):
        if eligible is not None:
            self.security_eligible = eligible
        self._share = share

    def take_vector_share(self, name):
        if self._share is None:
            raise MaterialStall(f"VECTOR_SHARE_MISSING: no vector share named {name!r}")
        return self._share


def test_party_refuses_a_dealer_provided_vector_under_an_eligible_dealer():
    for front in (_Front(True), _Front(None)):                 # eligible, or unknown (treated as eligible)
        p = Party(0, None, front, C, None)
        for optional in (False, True):
            with pytest.raises(MaterialError) as ei:
                p.vector_share("ln.g", optional=optional)
            assert ei.value.code == "VECTOR_SHARE_NOT_PRESHARED"
        assert "ln.g" not in p._vectors                         # nothing cached
        # a vector that was shared online (share_input stores it under the name) is used as before
        p._vectors["lin"] = np.arange(2, dtype=np.uint64)
        assert p.vector_share("lin").tolist() == [0, 1]
    # the insecure harness keeps its fallback
    p = Party(0, None, _Front(False), C, None)
    assert p.vector_share("ln.g").tolist() == V.tolist()
    # a missing vector is still a typed error (or None when optional), never a fallback
    p = Party(0, None, _Front(True, share=None), C, None)
    assert p.vector_share("lin.b", optional=True) is None
    with pytest.raises(MaterialError) as ei:
        p.vector_share("ln.g2")
    assert ei.value.code == "MATERIAL_MISSING_VECTOR"
