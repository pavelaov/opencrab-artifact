"""make_tables tests: the eight paper tables, status wrapping, bare tabulars, tolerant input (plain or gzip
compressed raw files, LF output), paired statistics, the header figures the prose cites (ratio per matrix, output
match lines, Student t intervals, identical-plan flags, noise floor, the derived modeled differences, the web
thread-cap grouping and warm paths, the like-with-like block ratio), the full12 rows of the dynamic table (after the
prefix rows, paired within their own matrix, a hyphen for a policy without a session), the marking line, and the
computed expectations: a cell without a measured record is an est-wrapped number, never a placeholder text, and a
measured record always prints plain."""

from __future__ import annotations

import json
import re

import pytest

from opencrab2.experiments import make_tables as mt
from opencrab2.experiments import schema
from opencrab2.highlevel.statistics import paired_bootstrap_ci


def _write(tmp_path, recs, name):
    for r in recs:
        schema.append_jsonl(tmp_path / "raw" / f"{name}.jsonl", r)


def _session(matrix, family, variant, policy, seed, wall, *, party=1, status="MEASURED_NATIVE", switches=0, aborts=0,
             warm=False, workers=2, match=True, n_layers=4, plans=None, mismatch=None):
    extra = {"matrix": matrix, "party": party, "session_wall_s": wall, "concurrent_workers": workers,
             "threads_per_process": 2, "warm_start": warm, "completed": aborts == 0,
             "T_warm_basis": "measured-warm-request" if warm else "derived-cold-minus-setup-install"}
    if plans is not None:
        extra["plans"] = list(plans)                       # the plan key of every epoch, as the runner records it
    if mismatch is not None:                               # (first differing step, reference margin) of a mismatch
        extra["first_mismatch_step"], extra["first_mismatch_margin"] = mismatch
    return schema.new_record(
        status=status, record_kind="session", workload_scope="prefix-4L" if n_layers == 4 else "full-model",
        clock_basis="wall-clock-exogenous", dealer_mode="ISOLATED_RECIPIENT_BUNDLES", trace_id=f"{family}:{variant}:{seed}",
        trace_hash="h" * 64, policy_id=policy, run_id=f"session:{matrix}:{family}:{variant}:{policy}:s{seed}:p{party}",
        repeat=seed, seed=seed, family=family, variant=variant, T=8, n_layers=n_layers, post_init_switches=switches,
        aborts=aborts, T_cold_s=None if warm else wall, T_warm_s=wall if warm else wall * 0.8, T_online_s=wall * 0.6,
        bytes=3 * 2 ** 20, rounds=100, material_consumed=5 * 2 ** 20, peak_rss_client=2 ** 30, peak_rss_server=2 ** 31,
        tokens=8, output_match=match, extra=extra)


def _sim(policy, family, variant, seed, total, kind="simulation", extra=None, T=8, n_layers=12, switches=0):
    e = {"epochs": [{"epoch": 0}], "final_plan": {}}
    e.update(extra or {})
    return schema.new_record(status="MODELED", record_kind=kind, workload_scope="full-model", clock_basis="modeled-epoch",
                             dealer_mode="NONE", trace_id=f"{family}:{variant}:seed{seed}", trace_hash="h" * 64,
                             policy_id=policy, run_id=f"{kind}:{policy}:{family}:{variant}:{seed}:{T}", repeat=seed,
                             T_cold_s=total, T_warm_s=total * 0.9, T_online_s=total * 0.8, seed=seed, family=family,
                             variant=variant, T=T, n_layers=n_layers, post_init_switches=switches, extra=e)


def _data_lines(text):
    body = text.split("\\midrule")[1].split("\\bottomrule")[0]
    return [l for l in body.splitlines() if l.strip()]


def _cells(line):
    return [c.strip() for c in line.rstrip().rstrip("\\").split("&")]


def test_wrap_weakest_and_fmt():
    assert mt.wrap("1.5", "MODELED") == "\\est{1.5}"
    assert mt.wrap("1.5", "ESTIMATED") == "\\est{1.5}"
    assert mt.wrap("1.5", "EMULATED_NUMERIC") == "\\est{1.5}"
    assert mt.wrap("1.5", "HISTORICAL") == "\\hist{1.5}"
    assert mt.wrap("1.5", "MEASURED_NATIVE") == "1.5"
    assert mt.wrap("1.5", "DIAGNOSTIC") == "\\diag{1.5}"
    assert mt.wrap("1.5", "EXTERNAL_REPORTED") == "\\ext{1.5}"
    assert mt.wrap("label", None) == "label"
    assert mt.weakest(["MEASURED_NATIVE", "HISTORICAL"]) == "HISTORICAL"
    assert mt.weakest(["MEASURED_NATIVE", "MODELED", "HISTORICAL"]) == "MODELED"
    assert mt.fmt(None) == "--" and mt.fmt(True) == "yes" and mt.fmt(0.0) == "0"
    assert mt.fmt(1234567) == "1234567" and "\\" not in mt.fmt(1234567)        # plain digits, no TeX macro
    assert mt.fmt(float("nan")) == "--" and mt.fmt(0.123456) == "0.123" and mt.fmt(1010.4) == "1010"
    assert "e" in mt.fmt(1.5e-7)


PLACEHOLDERS = ("not run", "n/a")


def _inner(cell):
    """The text of an expected (or modeled) cell: it must be wrapped in est and hold a computed value."""
    assert cell.startswith("\\est{") and cell.endswith("}"), cell
    inner = cell[len("\\est{"):-1]
    assert re.fullmatch(r"[-+0-9.e/ \[\],]+", inner), cell          # numbers only: never a word
    assert "--" not in inner and any(ch.isdigit() for ch in inner), cell
    return inner


def _assert_no_placeholder(text, name=""):
    low = text.lower()
    for word in PLACEHOLDERS:
        assert word not in low, (name, word)


def _header(text):
    return text.split("\\begin{tabular}")[0]


def test_empty_tree_yields_all_eight_tables_with_computed_expected_cells(tmp_path):
    (tmp_path / "raw").mkdir()                                       # an empty raw directory: nothing was measured
    manifest = mt.make_all(tmp_path)
    assert manifest["tables"] == list(mt.TABLE_NAMES) and not manifest["errors"]
    names = sorted(p.name for p in (tmp_path / "tables").glob("*.tex"))
    assert names == sorted(f"{n}.tex" for n in mt.TABLE_NAMES)
    for n in mt.TABLE_NAMES:
        text = (tmp_path / "tables" / f"{n}.tex").read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0].startswith("% generated by opencrab2.experiments.make_tables")
        assert lines[1].startswith("% ")                                         # inputs and record counts
        assert "\\begin{tabular}" in text and "\\end{tabular}" in text and "\\toprule" in text and "\\bottomrule" in text
        assert "\\begin{table}" not in text and "\\caption" not in text and "\\label" not in text
        # only the four status macros (plus booktabs rules and the tabular environment) appear
        macros = set(re.findall(r"\\([A-Za-z]+)", text.split("\\midrule", 1)[1]))
        assert macros <= {"est", "hist", "diag", "ext", "bottomrule", "end"}, (n, macros)
        _assert_no_placeholder(text, n)                                          # comments included
        counts = mt.cell_counts(text)
        assert counts == manifest["cells"][n] and "% cells: " in _header(text)
        if n == "catalog":
            assert counts == {"expected": 0, "measured": 0, "other": 0}
            continue
        rows = [_cells(l) for l in _data_lines(text)]
        assert rows
        n_values = 0
        for r in rows:
            assert not r[0].startswith("\\est{")                                 # the label
            for c in r[1:]:
                if n == "dynamic" and r[0].startswith("12L ") and c == mt.NOT_RUN:
                    continue                    # B1 is not a policy of the full12 matrix: no value, no expectation
                _inner(c)                                                        # every value cell is an est expectation
                n_values += 1
        assert counts == {"expected": n_values, "measured": 0, "other": 0}, n
        # the header says, per value column, how many cells are expected and where an expected value comes from
        head = _header(text)
        columns = _cells(text.split("\\toprule")[1].split("\\midrule")[0].strip())
        for j, col in enumerate(columns[1:], start=2):
            line = next(l for l in head.splitlines() if l.startswith(f"% column {j} ({col}): "))
            filled = sum(1 for r in rows if r[j - 1] != mt.NOT_RUN)
            assert f"{filled} expected, 0 measured" in line and "an expected value is " in line, (n, col)
    cat = (tmp_path / "tables" / "catalog.tex").read_text(encoding="utf-8")
    assert "registry.REGISTRY" in cat and "linear\\_wmask" in cat and "\\diag{DIAGNOSTIC}" in cat
    dyn = (tmp_path / "tables" / "dynamic.tex").read_text(encoding="utf-8")
    assert "missing" in dyn.splitlines()[1]
    # the full12 rows follow the prefix rows: no record yet, so no star; B1 is not a policy of the full12 matrix and
    # prints a hyphen, every planned policy (B2 on S1 included: nothing ran there yet) an expectation
    drows = [_cells(l) for l in _data_lines(dyn)]
    assert [r[0] for r in drows] == [lb for _f, _v, lb in mt.DIAG4_CELLS + mt.FULL12_CELLS]
    for r in drows[-len(mt.FULL12_CELLS):]:
        assert r[2] == "-" and _inner(r[6]).startswith("- / ") and all(_inner(c) for c in r[3:6]), r
    # planned counts: three sessions per cell, every one matching; thirty rejection probes per kind
    e2e = {_cells(l)[0]: _cells(l) for l in _data_lines((tmp_path / "tables" / "e2e.tex").read_text(encoding="utf-8"))}
    assert e2e["full 12L, B4"][8] == "\\est{3/3}" and e2e["full 12L, B4"][9] == "\\est{3}"
    web = {_cells(l)[0]: _cells(l) for l in _data_lines((tmp_path / "tables" / "web.tex").read_text(encoding="utf-8"))}
    assert web["GPT-2 1L"][7:11] == ["\\est{30/30}"] * 4 and web["GPT-2 1L"][4] == "\\est{1}" and web["tiny"][11] == "\\est{3}"
    # expected seconds are the dry-run predictor's: cold = warm + installation, the full model above the prefix
    f12, d4 = e2e["full 12L, B4"], e2e["prefix 4L, B4"]
    assert float(_inner(f12[1])) > float(_inner(f12[2])) > float(_inner(f12[3])) > 0.0
    assert float(_inner(f12[1])) > float(_inner(d4[1])) and float(_inner(f12[6])) > float(_inner(d4[6]))


def test_expected_session_values_come_from_the_predictor_and_the_formulas(tmp_path):
    from opencrab2.experiments import run_sessions as rs
    from opencrab2.highlevel import candidates as cands

    ex = mt.Expectations(tmp_path)
    cell = rs.Cell("diag4", "S0", "stable-LAN", "B0_fixed:wmask", 1, 4, 8)
    pred = rs.predict_cell(cell, ex.model, ex.rate)
    got = ex.session("S0", "stable-LAN", "B0_fixed:wmask", 4, 8)
    assert got["plan"] == rs.PLAN_WMASK
    assert got["T_cold_s"] == pytest.approx(pred["predicted_total_s"]) and got["T_online_s"] == pytest.approx(pred["online_s"])
    assert got["T_warm_s"] == pytest.approx(pred["predicted_total_s"] - pred["install_s"])
    wl, shapes_for, catalog = mt._workload(cell)
    online = [cands.plan_bytes_rounds(rs.PLAN_WMASK, shapes_for(e)) for e in range(cell.T)]
    assert got["bytes"] > sum(b for b, _r in online) and got["rounds"] > sum(r for _b, r in online)   # plus installation
    assert got["material"] >= pred["material_bytes_per_party"]
    warm = ex.session("S8", "warm-all-LAN", "B0_fixed:wmask", 4, 8)          # installed: no installation anywhere
    assert warm["total_s"] == pytest.approx(warm["T_warm_s"]) and warm["bytes"] < got["bytes"]
    assert ex.session("S0", "stable-LAN", "B2_remaining_plus_install", 4, 8) is ex.session("S0", "stable-LAN", "B4_opencrab", 4, 8)


def test_catalog_comes_from_the_registry():
    from opencrab2.highlevel import registry
    text = mt.catalog_table()
    rows = _data_lines(text)
    assert len(rows) == len(registry.REGISTRY)
    for line, comp in zip(rows, registry.REGISTRY.values()):
        c = _cells(line)
        assert c[0] == mt.tex_escape(comp.name) and c[2] == mt.tex_escape(comp.status)
        assert c[3] == {"ADMITTED": "MEASURED\\_NATIVE", "TEST_ONLY": "\\diag{DIAGNOSTIC}"}.get(comp.status, "none")


def test_blocks_table_one_row_per_block_measured_plain_and_expected_est(tmp_path):
    def prim(component, batch, per, status="MEASURED_NATIVE", params=None, nbytes=0, rounds=0, lang=None):
        extra = {"component": component, "batch": batch, "median_s": per * batch, "per_element_s": per,
                 "params": params or {}}
        if lang:
            extra["language"] = lang
            extra["layer"] = "lowlevel"
        return schema.new_record(status=status, record_kind="primitive", workload_scope="primitive",
                                 clock_basis="process-perf-counter",
                                 run_id=f"p:{component}:{batch}:{lang}:{status}:{(params or {}).get('step', '')}",
                                 bytes=nbytes, rounds=rounds, extra=extra)
    _write(tmp_path, [prim("drelu", 1024, 3e-6, params={"with_product": True}, nbytes=16 * 1024, rounds=1),
                      prim("drelu", 4096, 2e-6, params={"with_product": True}, nbytes=16 * 4096, rounds=1),
                      prim("toy_bridge", 512, 9e-6, params={"step": "toy_keygen", "N": 512}, nbytes=8192),
                      prim("toy_bridge", 512, 5e-6, params={"step": "toy_encrypt", "N": 512}, nbytes=8192),
                      prim("toy_bridge", 512, 1e-6, params={"step": "toy_decrypt", "N": 512}, nbytes=8192),
                      prim("trunc_faithful", 1024, 2e-6, params={"L": 40, "f": 16}, nbytes=8 * 1024, rounds=1),
                      prim("dcf_eval", 4096, 1e-6, status="HISTORICAL")], "primitives")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "blocks.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    lines = [_cells(l) for l in _data_lines(text)]
    assert [r[0] for r in lines] == [shown for _key, shown in mt.BLOCKS] and len(lines) == 9      # one row per block
    assert all(len(r) == 5 for r in lines)
    rows = {r[0]: r for r in lines}
    d = rows["DReLU, two payloads"]
    # the largest batch; bytes per element and party; a measured record prints plain; C++ expected = Python x ratio (1)
    assert d[1] == "8" and d[2] == "1" and d[3] == "2" and d[4] == "\\est{2}"
    assert rows["toy bridge H2A-ref"][1:4] == ["\\diag{8}", "\\diag{0}", "\\diag{6}"]             # keygen is not a step
    assert all(c.startswith("\\est{") for c in rows["DReLU, one payload"][1:])
    assert "cpp_bench.jsonl: missing" in text and "dcf\\_eval" not in text
    assert "toy bridge row: Python cell = toy_encrypt + toy_decrypt of toy_bridge at N = 512 (batch 512); C++ cell " \
           "expected (no record); the row enters no ratio" in text
    assert "the primitive records carry security_eligible false: in-process timings" in text
    assert mt.cell_counts(text) == {"expected": 27, "measured": 6, "other": 3}
    _write(tmp_path, [prim("drelu_p2", 4096, 4e-6, lang="cpp", nbytes=16 * 4096, rounds=1),
                      prim("drelu_p2_material", 4096, 9e-7, lang="cpp"),
                      prim("drelu_p1", 4096, 3e-6, lang="cpp", nbytes=16 * 4096, rounds=1),
                      # The row is the argmax over the vocabulary: only a record of that size fills it.
                      prim("argmax_k4", mt.VOCAB, 4e-5, status="DIAGNOSTIC", lang="cpp", nbytes=4 * mt.VOCAB, rounds=8),
                      # A record of another size must be ignored (it once stood in for the row).
                      prim("argmax_k4", 256, 9e-5, status="DIAGNOSTIC", lang="cpp", nbytes=99 * 256, rounds=3),
                      prim("argmax_k8", 4096, 9e-5, status="DIAGNOSTIC", lang="cpp", nbytes=99 * 4096, rounds=3),
                      # Unlike parameters (another truncation length): measured in both languages, outside the ratio.
                      prim("trunc_faithful", 1024, 8e-6, lang="cpp", params={"L": 32, "f": 16}, nbytes=8 * 1024, rounds=1),
                      # The local toy procedure at another N is the C++ toy cell; the two-party reference is not the row.
                      prim("toy_bridge_local", 128, 6e-6, lang="cpp", params={"N": 128}, nbytes=256, rounds=1),
                      prim("toy_bridge_h2a_ref", 128, 9e-5, lang="cpp", params={"N": 128}, nbytes=9999, rounds=3)],
           "cpp_bench")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "blocks.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    assert rows["DReLU, two payloads"][3:] == ["2", "4"]                       # both measured alike: the ratio is 2
    assert "ratio used for expectations: 2 (median over the 1 blocks measured alike in both languages: DReLU, two " \
           "payloads); measured in both languages but outside the ratio: TruncFaithful (parameters differ: Python " \
           "L = 40, f = 16; C++ L = 32, f = 16); toy bridge H2A-ref (test-only procedure)" in text
    assert rows["DReLU, one payload"] == ["DReLU, one payload", "8", "1", "\\est{1.5}", "3"]      # Python = C++ / ratio
    assert rows["argmax over 50257, k = 4"][1:] == ["\\diag{2}", "\\diag{8}", "\\est{20}", "\\diag{40}"]
    assert all(c.startswith("\\est{") for c in rows["argmax over 50257, k = 8"][1:])   # the n = 4096 record is not this row
    assert rows["TruncFaithful"][1:] == ["4", "1", "2", "8"]                   # both measured, printed, not in the ratio
    assert rows["toy bridge H2A-ref"][3:] == ["\\diag{6}", "\\diag{6}"]        # the local variant, not the h2a reference
    assert "C++ cell = toy_bridge_local at N = 128 (batch 128); the row enters no ratio" in text
    assert "toy_bridge_h2a_ref is another procedure without a Python counterpart and is not printed" in text


def test_costmodel_table_stale_until_the_formulas_hash_matches(tmp_path):
    (tmp_path / "calibration").mkdir()
    t = {"n": 10, "mape": 0.12, "median_rel_err": 0.03, "p95_rel_err": 0.4, "spearman": 0.99, "top1_lane_acc": 0.9,
         "plan_identity": 1.0, "realized_regret_s": 0.01, "informative_pair_correct": 3, "informative_pair_decisions": 4,
         "coverage_2sigma": 0.8}
    m = {"models": {"fitted_MEASURED_NATIVE": {"test": t}, "handwritten_MODELED": {"test": dict(t, mape=0.3)}},
         "n_train": 20, "n_test": 10, "split_rule": {"holdout_shapes": [[1, 1]], "holdout_network": "WAN-mid", "holdout_n": 32},
         "campaign": {}}
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    text = mt.costmodel_table(tmp_path)
    _assert_no_placeholder(text)
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    assert "STALE" in text and rows["fitted"][2] == "\\est{0.12}" and rows["fitted"][9] == "\\est{3/4}"
    # the row the calibration lacks is an expectation computed from the fitted row
    assert rows["uncertainty-aware"][1:] == rows["fitted"][1:]
    # a metric the stale file leaves empty is computed (self-check of the two models), never printed as a dash
    m["models"]["handwritten_MODELED"]["test"]["coverage_2sigma"] = None
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    hand = {_cells(l)[0]: _cells(l) for l in _data_lines(mt.costmodel_table(tmp_path))}["hand-written"]
    assert 0.0 <= float(_inner(hand[10])) <= 1.0 and hand[2] == "\\est{0.3}"
    m["models"]["handwritten_MODELED"]["test"]["coverage_2sigma"] = 0.8
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    assert mt.cell_counts(text) == {"expected": 30, "measured": 0, "other": 0}
    import hashlib
    from opencrab2.highlevel import costmodel as cm
    # the hash the table compares with is that of the shipped results/calibration/formulas.json, whatever the data root
    (tmp_path / "calibration" / "formulas.json").write_text('{"v": 1}', encoding="utf-8")
    m["campaign"]["formulas_sha256"] = hashlib.sha256(cm.formulas_path().read_bytes()).hexdigest()
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    text = mt.costmodel_table(tmp_path)
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    # the hand-written row is the measured evaluation of the constants on the same hold-out rows: plain, whatever
    # status label the file gives the model (MODELED here); the header says so
    assert "STALE" not in text and rows["fitted"][2] == "0.12" and rows["hand-written"][2] == "0.3"
    assert "measured evaluation on the hold-out rows (model label in the file: MODELED)" in _header(text)
    assert rows["uncertainty-aware"][2] == "\\est{0.12}" and rows["uncertainty-aware"][6] == "\\est{0.9}"
    assert mt.cell_counts(text) == {"expected": 10, "measured": 20, "other": 0}
    # a metric a current file leaves empty is undefined for that model and prints as a plain dash
    m["models"]["handwritten_MODELED"]["test"]["coverage_2sigma"] = None
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    hand = {_cells(l)[0]: _cells(l) for l in _data_lines(mt.costmodel_table(tmp_path))}["hand-written"]
    assert hand[10] == "--" and hand[2] == "0.3"
    m["models"]["handwritten_MODELED"]["test"]["coverage_2sigma"] = 0.8
    m["models"]["uncertainty_MEASURED_NATIVE"] = {"test": dict(t, top1_lane_acc=0.95)}
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    text = mt.costmodel_table(tmp_path)
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    assert "STALE" not in text
    assert rows["fitted"][2] == "0.12" and rows["hand-written"][2] == "0.3" and rows["uncertainty-aware"][6] == "0.95"
    assert mt.cell_counts(text) == {"expected": 0, "measured": 30, "other": 0}


def test_dynamic_table_pairs_within_seed_and_reports_percent_with_interval(tmp_path):
    recs = []
    b4 = {1: 90.0, 2: 99.0, 3: 110.0}
    W, G = "embedding=wmask|linear=wmask", "embedding=generic|linear=generic"
    b4_plans = {1: [W, W], 2: [G, W], 3: [W, W]}             # seed 2 ran the prefill on the generic plan
    for seed in (1, 2, 3):
        recs.append(_session("diag4", "S1", "ramp-WAN-c-to-LAN", "B0_fixed:wmask", seed, 100.0, plans=[W, W]))
        recs.append(_session("diag4", "S1", "ramp-WAN-c-to-LAN", "B0_fixed:generic", seed, 150.0, plans=[G, G]))
        recs.append(_session("diag4", "S1", "ramp-WAN-c-to-LAN", "B4_opencrab", seed, b4[seed], switches=1,
                             plans=b4_plans[seed], match=seed != 2, mismatch=(3, 0.5) if seed == 2 else None))
        recs.append(_session("diag4", "S1", "ramp-WAN-c-to-LAN", "B4_opencrab", seed, 1.0, party=0))     # P0 copy ignored
        recs.append(_session("diag4", "S1", "ramp-WAN-c-to-LAN", "B2_remaining_plus_install", seed, 105.0, plans=[W, W]))
        # a cell where only the all-wmask fixed plan ran: B0 is that plan, the row is marked
        recs.append(_session("diag4", "S2", "rtt-step-up", "B0_fixed:wmask", seed, 200.0, plans=[W, W]))
        recs.append(_session("diag4", "S2", "rtt-step-up", "B4_opencrab", seed, 204.0, plans=[W, W]))
    recs.append(_session("diag4", "S1", "ramp-WAN-c-to-LAN", "B1_naive", 1, 500.0, aborts=1))          # aborted: excluded
    recs.append(_session("diag4", "S1", "ramp-WAN-c-to-LAN", "B1_naive", 2, 120.0, workers=4))         # other concurrency
    recs.append(_session("smoke", "S0", "stable-LAN", "B4_opencrab", 1, 5.0, match=False))              # pipeline matrix
    _write(tmp_path, recs, "sessions")
    (tmp_path / "raw" / "sessions.jsonl").open("a", encoding="utf-8").write("{not json\n")
    # the modeled campaign has the S0 stable-LAN cell at the matrix's T and layer count: its paired differences
    # and switch counts are the expectation of that cell; a 12-layer record of the same trace is not
    sim = []
    for seed in (1, 2, 3):
        sim.append(_sim("B0_fixed_best_hindsight", "S0", "stable-LAN", seed, 100.0, n_layers=4))
        sim.append(_sim("B4_opencrab", "S0", "stable-LAN", seed, 95.0 + seed, n_layers=4, switches=2))
        sim.append(_sim("B0_fixed_best_hindsight", "S0", "stable-WAN", seed, 100.0, n_layers=12))
        sim.append(_sim("B4_opencrab", "S0", "stable-WAN", seed, 50.0, n_layers=12, switches=2))
    _write(tmp_path, sim, "simulation")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "dynamic.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    assert "1 malformed lines skipped" in text and "1 aborted or incomplete sessions excluded" in text
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    r = rows["S1 ramp"]                                                          # both fixed plans ran: no marker
    assert r[1] == "100"                                                         # the cheaper fixed plan is B0
    mean, lo, hi = paired_bootstrap_ci([-10.0, -1.0, 10.0])
    assert r[4] == mt.fmt_ci(mean, lo, hi) and "\\est" not in r[4]               # a measured cell stays plain
    assert r[3] == mt.fmt_ci(5.0, 5.0, 5.0)
    _inner(r[2])                                                                 # B1 has no valid pair: expected
    assert r[5] == "3"
    sw = _inner(r[6]).split(" / ")                                               # one policy unmeasured: est as a whole
    assert len(sw) == 3 and sw[1:] == ["0", "1"]
    s2 = rows["S2 RTT only*"]                                                    # only the all-wmask plan ran: marked
    assert s2[1] == "200" and s2[4] == mt.fmt_ci(2.0, 2.0, 2.0) and s2[5] == "3"
    for label in ("S0 stable WAN", "S3 two changes", "S8 cold", "S8 warm installed"):
        vals = [_inner(c) for c in rows[label][1:]]
        assert vals[4] == "3" and "[" not in vals[1], label                      # planned n; the predictor has no interval
        assert vals[1] == vals[2] == vals[3]                                     # it prices B1, B2 and B4 alike
    s0 = rows["S0 stable LAN"]
    mean, lo, hi = paired_bootstrap_ci([-4.0, -3.0, -2.0])
    assert s0[4] == "\\est{" + mt.fmt_ci(mean, lo, hi) + "}"                     # from the modeled records of the cell
    assert _inner(s0[6]).split(" / ")[2] == "2" and "[" not in _inner(s0[2])
    assert float(_inner(rows["S8 cold"][1])) > float(_inner(rows["S8 warm installed"][1]))   # installation
    # 35 expected cells in the prefix rows plus 5 per full12 row without records (B1 prints a hyphen, not counted)
    assert mt.cell_counts(text) == {"expected": 45, "measured": 7, "other": 0}
    head = _header(text)
    # the concurrency level: the majority level is kept, the stray 4-worker record is left out and counted
    assert "concurrency level 2 concurrent sessions, 2 threads per process; 1 completed records of another concurrency " \
           "level left out (levels are never pooled)" in head
    # the marked row and its fixed plan
    assert "so B0 is that plan and not the better of two: S2 RTT only* (all-wmask)" in head
    # per printed difference: the Student t interval and the identical-plan count; B2 ran B0's plans in every seed,
    # B4 in two of three (the generic prefill of seed 2)
    st = mt.student_t_ci([-10.0, -1.0, 10.0])
    assert "per printed difference, one line per cell: mean, Student t interval (2 degrees of freedom at n = 3)" in head
    assert ("S1 ramp: B2 5, Student t [5, 5], identical plan sequence as B0 in 3 of 3 seeds; B4 "
            f"{mt.fmt(st[0])}, Student t [{mt.fmt(st[1])}, {mt.fmt(st[2])}], identical plan sequence as B0 in 2 of 3 seeds") in head
    assert "S2 RTT only*: B4 2, Student t [2, 2], identical plan sequence as B0 in 3 of 3 seeds" in head
    # the noise floor over the identical-plan pairs of the matrix: B2 (5, 5, 5), B4 seeds 1 and 3 (10, 10), S2 (2, 2, 2)
    assert "noise floor: largest absolute paired difference among the 8 identical-plan pairs of the matrix" in head
    assert ": 10 percent (median 5)" in head
    # output match per matrix, a mismatch with its step and margin, one without the fields; smoke is a pipeline check
    assert "output match diag4: 18 of 19 completed sessions match the float reference" in head
    assert "mismatch session:diag4:S1:ramp-WAN-c-to-LAN:B4_opencrab:s2:p1: first differing step 3, reference margin 0.5" in head
    assert "output match smoke (pipeline check, evidence for no hypothesis): 0 of 1 completed sessions match" in head
    assert "mismatch session:smoke:S0:stable-LAN:B4_opencrab:s1:p1: first differing step and reference margin not in the " \
           "record (the fields first_mismatch_step and first_mismatch_margin are absent)" in head
    # the ratio: campaign sessions only, per matrix, with the concurrency named; the smoke session is excluded
    assert "per matrix: diag4 (2 concurrent sessions, 2 threads per process or 4 concurrent sessions, 2 threads per " \
           "process): median" in head and "n = 19; 1 completed session of the pipeline matrices (smoke) excluded" in head
    assert "any matrix took" not in head


def test_dynamic_table_prints_the_full12_cells_after_the_prefix_rows(tmp_path):
    W, G = "argmax_k=2|embedding=wmask|linear=wmask|max_k=2", "argmax_k=2|embedding=generic|linear=generic|max_k=2"
    W4 = "argmax_k=2|embedding=wmask|linear=wmask|max_k=4"
    b0, b2, b4 = {1: 400.0, 2: 320.0, 3: 300.0}, {1: 404.0, 2: 320.0, 3: 297.0}, {1: 408.0, 2: 323.2, 3: 306.0}
    r0, r4 = {1: 1000.0, 2: 1250.0, 3: 1200.0}, {1: 980.0, 2: 1225.0, 3: 1164.0}
    recs = []
    for seed in (1, 2, 3):
        # the prefix cell of the same trace (diag4, 2 concurrent sessions): both fixed plans ran, B0 = all-wmask
        recs.append(_session("diag4", "S0", "stable-LAN", "B0_fixed:wmask", seed, 100.0, plans=[W, W]))
        recs.append(_session("diag4", "S0", "stable-LAN", "B0_fixed:generic", seed, 150.0, plans=[G, G]))
        recs.append(_session("diag4", "S0", "stable-LAN", "B4_opencrab", seed, 110.0, plans=[W, W]))
        # full12 (3 concurrent sessions): S0 with B0 all-wmask, B2 and B4 on B0's plans; S1 with B0 and B4 only,
        # B4 on the max_k = 4 plan and one switch
        full = dict(n_layers=12, workers=3)
        recs.append(_session("full12", "S0", "stable-LAN", "B0_fixed:wmask", seed, b0[seed], plans=[W, W], **full))
        recs.append(_session("full12", "S0", "stable-LAN", "B2_remaining_plus_install", seed, b2[seed], plans=[W, W],
                             **full))
        recs.append(_session("full12", "S0", "stable-LAN", "B4_opencrab", seed, b4[seed], plans=[W, W], **full))
        recs.append(_session("full12", "S0", "stable-LAN", "B4_opencrab", seed, 1.0, party=0, **full))  # P0 copy ignored
        recs.append(_session("full12", "S1", "ramp-WAN-c-to-LAN", "B0_fixed:wmask", seed, r0[seed], plans=[W, W], **full))
        recs.append(_session("full12", "S1", "ramp-WAN-c-to-LAN", "B4_opencrab", seed, r4[seed], plans=[W4, W4],
                             switches=1, **full))
    _write(tmp_path, recs, "sessions")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "dynamic.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    lines = _data_lines(text)
    labels = [_cells(l)[0] for l in lines]
    # the full12 rows come after the seven prefix rows; only all-wmask ran there, so both carry the star
    assert labels == [lb for _f, _v, lb in mt.DIAG4_CELLS] + ["12L S0 stable LAN*", "12L S1 ramp*"]
    rows = {_cells(l)[0]: _cells(l) for l in lines}

    def diffs(pol, ref):
        return [100.0 * (pol[s] - ref[s]) / ref[s] for s in (1, 2, 3)]

    def ci(d):
        return mt.fmt_ci(*paired_bootstrap_ci(d))

    # pairs never cross matrices: the prefix B0 is the prefix all-wmask mean, the full12 B0 the full12 one
    assert rows["S0 stable LAN"][1] == "100" and rows["S0 stable LAN"][4] == ci([10.0, 10.0, 10.0])
    f0 = rows["12L S0 stable LAN*"]
    assert f0 == ["12L S0 stable LAN*", "340", "-", ci(diffs(b2, b0)), ci(diffs(b4, b0)), "3", "- / 0 / 0"]
    f1 = rows["12L S1 ramp*"]
    assert f1 == ["12L S1 ramp*", "1150", "-", "-", ci(diffs(r4, r0)), "3", "- / - / 1"]
    assert not any("--" in l for l in lines[-2:]) and "\\est" not in "".join(lines[-2:])     # measured: plain
    assert mt.cell_counts(text) == {"expected": 39, "measured": 12, "other": 0}
    head = _header(text)
    # the inputs line counts both matrices, each at its own concurrency level (nothing left out: never pooled)
    assert ("inputs: raw/sessions.jsonl: 27 records, 24 used; matrix diag4, client (P1) records, 9 used; 0 aborted or "
            "incomplete sessions excluded; concurrency level 2 concurrent sessions, 2 threads per process; matrix full12, "
            "client (P1) records, 15 used; 0 aborted or incomplete sessions excluded; concurrency level 3 concurrent "
            "sessions, 2 threads per process\n") in head
    # the prefix header lines know nothing of the full12 pairs
    assert "* = only one fixed plan ran in the cell (the other" not in head
    assert "% S0 stable LAN: B4 10, Student t [10, 10], identical plan sequence as B0 in 3 of 3 seeds\n" in head
    assert "% noise floor: largest absolute paired difference among the 3 identical-plan pairs of the matrix (a policy " \
           "session with the plan sequence of its seed's B0 session): 10 percent (median 10)\n" in head
    # the full12 lines: star cells, hyphens, Student t per cell with the identical-plan flag, own noise floor
    assert ("% full12 rows (label prefix 12L; 12 layers): B0 and the paired differences are formed within the full12 "
            "matrix by the rules above, never across matrices") in head
    assert ("so B0 is that plan and not the better of two: 12L S0 stable LAN* (all-wmask; the matrix plans no other "
            "fixed plan); 12L S1 ramp* (all-wmask; the matrix plans no other fixed plan)\n") in head
    assert ("% full12 - = the policy has no session in the cell, printed without a value or an expectation: B1 in 12L S0 "
            "stable LAN* (not a policy of the matrix); B1 in 12L S1 ramp* (not a policy of the matrix); B2 in 12L S1 "
            "ramp* (a policy of the matrix without a session in the cell)\n") in head
    assert "% full12 per printed difference, one line per cell: mean, Student t interval (2 degrees of freedom at n = 3)" in head

    def student(d):
        st = mt.student_t_ci(d)
        return f"{mt.fmt(st[0])}, Student t [{mt.fmt(st[1])}, {mt.fmt(st[2])}]"

    assert (f"% 12L S0 stable LAN*: B2 {student(diffs(b2, b0))}, identical plan sequence as B0 in 3 of 3 seeds; "
            f"B4 {student(diffs(b4, b0))}, identical plan sequence as B0 in 3 of 3 seeds\n") in head
    st = mt.student_t_ci(diffs(r4, r0))
    assert st[2] < 0.0                                                   # the interval excludes zero, as in the campaign
    assert f"% 12L S1 ramp*: B4 {student(diffs(r4, r0))}, identical plan sequence as B0 in 0 of 3 seeds\n" in head
    # identical-plan pairs of full12: B2 and B4 on S0 (|1|, 0, |1|, 2, 1, 2); the max_k = 4 sessions of S1 are not
    assert "% full12 noise floor: largest absolute paired difference among the 6 identical-plan pairs of the full12 " \
           "matrix (a policy session with the plan sequence of its seed's B0 session): 2 percent (median 1)\n" in head
    assert "output match full12: 15 of 15 completed sessions match the float reference" in head


def test_modeled_and_factorial_tables_are_fully_est_wrapped(tmp_path):
    sim = []
    for seed in (1, 2, 3):
        for variant in ("a", "b"):
            sim.append(_sim("B0_fixed_best_hindsight", "S3", variant, seed, 100.0))
            sim.append(_sim("B4_opencrab", "S3", variant, seed, 95.0 + seed))
            sim.append(_sim("B2_remaining_plus_install", "S3", variant, seed, 101.0))
            sim.append(_sim("B1_naive", "S3", variant, seed, 110.0))
            sim.append(_sim("B4_no_install", "S3", variant, seed, 100.5))        # ablations: derived header figures
            sim.append(_sim("delayed:2", "S3", variant, seed, 164.0))
    _write(tmp_path, sim, "simulation")
    fact = []
    for seed in (1, 2):
        for planner in ("ExactAssign", "GreedyPerOperator"):
            for control in ("B2", "B4"):
                fact.append(_sim(f"F:{planner}x{control}", "S0", "a", seed, 90.0, kind="factorial",
                                 extra={"planner": planner, "control": control, "solver_latency_mean_s": 1e-4,
                                        "optimality_gap_mean_s": 0.0, "identical_to_exact_rate": 1.0}))
    _write(tmp_path, fact, "factorial")
    manifest = mt.make_all(tmp_path)
    assert manifest["records_by_status"]["MODELED"] == len(sim) + len(fact)
    text = (tmp_path / "tables" / "dynamic_modeled.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    r = rows["S3, T = 8"]
    assert all(c.startswith("\\est{") for c in r[1:]), r
    assert r[1] == "\\est{100}" and r[5] == "\\est{6}" and r[2].startswith("\\est{10 [")
    s0 = [_inner(c) for c in rows["S0, T = 8"][1:]]                              # a family without records: predictor
    assert float(s0[0]) > 0.0 and s0[4] == "40" and s0[5] == "0 / 0 / 0"
    assert manifest["cells"]["dynamic_modeled"] == {"expected": 48, "measured": 0, "other": 6}
    # the derived figures the prose cites: the ablations against B0 at 8 tokens, same pairing as the columns
    head = _header(text)
    assert "B4_no_install on S3 at T = 8: 0.5 [0.5, 0.5] (n = 6); delayed:2 on S3 at T = 8: 64 [64, 64] (n = 6)" in head
    assert "B4_no_install on S1 at T = 8: no campaign record; delayed:2 on S1 at T = 8: no campaign record" in head
    ftext = (tmp_path / "tables" / "factorial.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(ftext)
    frows = {_cells(l)[0]: _cells(l) for l in _data_lines(ftext)}
    assert set(frows) == {"ExactAssign", "GreedyPerOperator"}
    assert all(c.startswith("\\est{") for c in frows["ExactAssign"][1:]) and frows["ExactAssign"][6] == "\\est{4}"
    assert manifest["cells"]["factorial"] == {"expected": 0, "measured": 0, "other": 12}
    assert mt.write_factorial_table(tmp_path).name == "factorial.tex"


def test_e2e_table_measured_warm_plain_and_derived_warm_est(tmp_path):
    recs = []
    for seed in (1, 2, 3):
        recs.append(_session("diag4", "S0", "stable-LAN", "B4_opencrab", seed, 60.0))
        recs.append(_session("diag4", "S8", "warm-all-LAN", "B4_opencrab", seed, 40.0, warm=True))
        recs.append(_session("full12", "S0", "stable-LAN", "B4_opencrab", seed, 200.0, n_layers=12, match=(seed != 3)))
    _write(tmp_path, recs, "sessions")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "e2e.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    d = rows["prefix 4L, B4"]
    assert d[1] == "60" and d[2] == "40" and d[3] == "36" and d[4] == "3" and d[5] == "100" and d[6] == "5"
    assert d[7] == "2050" and d[8] == "3/3" and d[9] == "3"
    f = rows["full 12L, B4"]
    assert f[1] == "200" and f[2] == "\\est{160}" and f[8] == "2/3"            # derived warm time is not a measurement
    b2 = [_inner(c) for c in rows["full 12L, B2"][1:]]                         # no record: expected, never a word
    assert b2[7] == "3/3" and b2[8] == "3" and float(b2[0]) > float(b2[1]) > float(b2[2])
    head = _header(text)                                                       # measured against the predictor, per matrix
    assert "measured over predicted session seconds of the completed campaign sessions, per matrix: diag4 (2 concurrent " \
           "sessions, 2 threads per process): median" in head and "; full12 (2 concurrent sessions" in head
    assert "concurrency: diag4 rows at 2 concurrent sessions, 2 threads per process; full12 rows at 2 concurrent" in head
    assert "output match diag4: 6 of 6 completed sessions match the float reference" in head
    assert "output match full12: 2 of 3 completed sessions match the float reference" in head
    assert "mismatch session:full12:S0:stable-LAN:B4_opencrab:s3:p1: first differing step and reference margin not in " \
           "the record" in head
    assert mt.cell_counts(text) == {"expected": 36, "measured": 17, "other": 1}


def test_web_table_measured_cells_plain_rejections_and_missing_workloads(tmp_path):
    from opencrab2.experiments import run_web

    cw = {"ttft_cold_s": 2.5, "ttft_warm_s": 1.5, "warm_verified": True, "itl_mean_s": 0.4, "itl_p95_s": 0.6,
          "completion": 1.0, "tokens": 4, "first_T_cold_s": 5.0, "second_T_warm_s": 3.0, "first_T_online_s": 4.0,
          "first_bytes": 1000, "first_rounds": 20, "second_security_eligible": True, "aborted": False,
          "output_match": True, "token_ids_first": [1, 2, 3]}
    rj = {f"rejected_{k}_trials": 5 for k in run_web.REJECTION_KINDS}
    rj.update({f"rejected_{k}_accepted_wrongly": 0 for k in run_web.REJECTION_KINDS})
    rj["rejected_token_accepted_wrongly"] = 1
    rj["aborted"] = False
    cancel = {"cancel_waste_bytes": 2 * 2 ** 20, "completion": 0.25, "aborted": False}
    rc = {"recovery_s": 0.7, "completion": 1.0, "security_eligible_final": True, "aborted": False}
    recs = [run_web.web_record("cold_warm", 0, cw, 1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny"),
            run_web.web_record("cold_warm", 1, dict(cw, warm_verified=False, ttft_warm_s=None, second_security_eligible=False),
                               1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny"),
            run_web.web_record("rejections", 0, rj, 1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny"),
            run_web.web_record("cancel", 0, cancel, 1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny"),
            run_web.web_record("reconnect", 0, rc, 1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny"),
            run_web.web_record("cold_warm", 0, cw, 1, 4, "x", None, "gpt2@abc", "B4_opencrab", "gpt2-1L")]
    assert recs[0]["security_eligible"] is True and recs[1]["security_eligible"] is False
    assert recs[0]["T_cold_s"] == 5.0 and recs[0]["T_warm_s"] == 3.0 and recs[1]["T_warm_s"] is None
    assert recs[0]["output_match"] is True and "token_ids_first" not in recs[0]["extra"]
    assert recs[0]["run_id"] == run_web.web_run_id("tiny", "cold_warm", 1, 4, 0)
    _write(tmp_path, recs, "web")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "web.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    assert "\\hist{" not in text
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    t = rows["tiny"]
    # TTFT = mean [min, max] over the repeats (the warm range over the verified warm requests only)
    assert t[1] == "2.5 [2.5, 2.5]" and t[2] == "1.5 [1.5, 1.5]" and t[3] == "0.4" and t[4] == "1" and t[5] == "2"
    assert t[6] == "0.7" and t[7] == "5/5" and t[8] == "4/5" and t[11] == "2"
    g1 = rows["GPT-2 1L"]                                # one scenario measured: the other cells are expected
    assert g1[1:5] == ["2.5 [2.5, 2.5]", "1.5 [1.5, 1.5]", "0.4", "1"] and g1[11] == "1"
    assert float(_inner(g1[5])) > 0.0 and float(_inner(g1[6])) > 0.0 and g1[7:11] == ["\\est{30/30}"] * 4
    g4 = [_inner(c) for c in rows["GPT-2 4L"][1:]]
    assert g4[3] == "1" and g4[6:10] == ["30/30"] * 4 and g4[10] == "3"
    assert float(g4[0]) > float(g4[1]) > float(g4[2]) > 0.0      # cold TTFT > warm TTFT (prefill) > one decode step
    assert mt.cell_counts(text) == {"expected": 17, "measured": 16, "other": 0}


def test_web_table_never_pools_thread_caps_and_lists_the_warm_paths(tmp_path):
    from opencrab2.experiments import run_web

    plan_w, plan_g = "argmax_k=2|embedding=wmask|linear=wmask|max_k=2", "argmax_k=2|embedding=generic|linear=generic|max_k=2"

    def cw(cold, warm, plan=None, switched=None, reason=None):
        m = {"ttft_cold_s": cold, "ttft_warm_s": warm, "warm_verified": True, "itl_mean_s": 0.4, "completion": 1.0,
             "tokens": 4, "first_T_cold_s": 5.0, "second_T_warm_s": 3.0, "second_security_eligible": True, "aborted": False,
             "second_epoch0_plan_key": plan, "second_epoch0_switched": switched, "second_epoch0_decision_reason": reason}
        return m

    # an uncapped record first (no threads_per_process field), then two capped repeats with the epoch-0 path
    recs = [run_web.web_record("cold_warm", 2, cw(9.0, 8.0), 1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny",
                               threads_per_process=None),
            run_web.web_record("cold_warm", 0, cw(2.5, 1.5, plan_w, False, "keep"), 1, 4, "x", None, "tiny@abc",
                               "B4_opencrab", "tiny", threads_per_process=2),
            run_web.web_record("cold_warm", 1, cw(3.5, 2.5, plan_g, True, "install"), 1, 4, "x", None, "tiny@abc",
                               "B4_opencrab", "tiny", threads_per_process=2)]
    assert "threads_per_process" in recs[0]["extra"] and recs[0]["extra"]["threads_per_process"] is None
    _write(tmp_path, recs, "web")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "web.tex").read_text(encoding="utf-8")
    _assert_no_placeholder(text)
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    t = rows["tiny"]
    assert t[1] == "3 [2.5, 3.5]" and t[2] == "2 [1.5, 2.5]" and t[11] == "2"      # the uncapped record is left out
    head = _header(text)
    assert "tiny: thread cap 2 per process (3 co-located processes); left out: 1 records without the field; " \
           "repeat 0: TTFT cold 2.5 s, warm 1.5 s, warm request at epoch 0: plan " + plan_w + ", kept (keep); " \
           "repeat 1: TTFT cold 3.5 s, warm 2.5 s, warm request at epoch 0: plan " + plan_g + ", switched (install)" in head
    assert "cancel and rejections records carry security_eligible false" in head
    # a later record at another cap: its cap is used, the cap-2 records are left out and counted
    _write(tmp_path, [run_web.web_record("cold_warm", 3, cw(7.0, 6.0), 1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny",
                                         threads_per_process=4)], "web")
    mt.make_all(tmp_path)
    text = (tmp_path / "tables" / "web.tex").read_text(encoding="utf-8")
    rows = {_cells(l)[0]: _cells(l) for l in _data_lines(text)}
    assert rows["tiny"][1] == "7 [7, 7]" and rows["tiny"][11] == "1"
    assert "tiny: thread cap 4 per process (3 co-located processes); left out: 2 records cap 2, 1 records without the " \
           "field; repeat 3: TTFT cold 7 s, warm 6 s, warm request at epoch 0: plan path not in the record (the epoch0 " \
           "fields are absent)" in _header(text)


def test_compressed_raw_files_are_read_and_outputs_use_lf_and_carry_the_marking_line(tmp_path):
    import gzip

    recs = [_session("full12", "S0", "stable-LAN", "B4_opencrab", seed, 200.0, n_layers=12) for seed in (1, 2, 3)]
    plain = tmp_path / "plain.jsonl"
    for r in recs:
        schema.append_jsonl(plain, r)
    (tmp_path / "raw").mkdir()
    with gzip.open(tmp_path / "raw" / "sessions.jsonl.gz", "wb") as fh:          # only the compressed file exists
        fh.write(plain.read_bytes())
    manifest = mt.make_all(tmp_path)
    assert manifest["inputs"]["sessions"] == {"present": True, "records": 3, "malformed": 0}
    e2e = (tmp_path / "tables" / "e2e.tex").read_text(encoding="utf-8")
    assert "inputs: raw/sessions.jsonl.gz: 3 records, 3 used" in e2e
    assert {_cells(l)[0]: _cells(l) for l in _data_lines(e2e)}["full 12L, B4"][1] == "200"
    for p in list((tmp_path / "tables").glob("*.tex")) + [tmp_path / "tables" / "manifest.json"]:
        raw = p.read_bytes()
        assert b"\r\n" not in raw and raw.endswith(b"\n"), p.name                  # LF endings on every platform
    for n in mt.TABLE_NAMES:
        text = (tmp_path / "tables" / f"{n}.tex").read_text(encoding="utf-8")
        marking = [l for l in _header(text).splitlines() if l.startswith("% marking: ")]
        assert marking == ["% " + mt.MARKING], n                                   # one marking line per table
    # a Student t interval: mean +- t(0.975, n - 1) x sample sd / sqrt(n); none below two values
    import math
    mean, lo, hi = mt.student_t_ci([1.0, 2.0, 3.0])
    half = 4.302652729911275 / math.sqrt(3.0)
    assert mean == 2.0 and lo == pytest.approx(2.0 - half) and hi == pytest.approx(2.0 + half)
    assert mt.student_t_ci([1.0]) is None and mt.fmt_range([2.0, 1.0, 3.0]) == "2 [1, 3]"


def test_obsolete_generated_tables_are_moved_aside(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "policy_T8.tex").write_text("% generated by opencrab2.experiments.make_tables from experiments/raw\n", encoding="utf-8")
    (tables / "handmade.tex").write_text("% written by hand\n", encoding="utf-8")
    manifest = mt.make_all(tmp_path)
    assert manifest["retired"] == ["policy_T8.tex"]
    assert (tables / "obsolete" / "policy_T8.tex").is_file() and (tables / "handmade.tex").is_file()


def test_run_web_command_lines_match_the_platform_parsers(tmp_path):
    """The planned command lines must be accepted by the platform CLIs (flags, host:port types)."""
    from opencrab2.experiments import run_web
    pytest.importorskip("aiohttp")
    from opencrab2.platform import dealer_service, model_server, sidecar
    st = run_web.Stack(tmp_path, "npz:tiny.npz", 1, trace={"trace_id": "t", "trace_hash": "h", "points": [[0.0, 1e9, 0.0, 1.0, 0.0]]},
                       model_version="tiny@abc")
    st.dealer_port, st.server_port = 40001, 40002
    cmds = st.commands()
    for name, parser in (("dealer", dealer_service.build_parser()), ("server", model_server.build_parser()),
                         ("sidecar", sidecar.build_parser())):
        argv = cmds[name][3:]
        assert cmds[name][:2] == [run_web.PYTHON, "-m"] and cmds[name][2].startswith("opencrab2.platform.")
        ns = parser.parse_args(argv)
        assert ns is not None
    ns = model_server.build_parser().parse_args(cmds["server"][3:])
    assert ns.dealer == ("127.0.0.1", 40001) and ns.ws == ("127.0.0.1", 40002) and ns.model_version == "tiny@abc"
    assert ns.dealer_key_file.endswith("key0.hex") and ns.trace_json.endswith("trace.json")
    ns = sidecar.build_parser().parse_args(cmds["sidecar"][3:])
    assert ns.server == "ws://127.0.0.1:40002/session" and ns.dealer_key_file.endswith("key1.hex")
    assert ns.token_file.endswith("token.txt") and ns.length_bucket == 32 and ns.client_key_domain == "default"
