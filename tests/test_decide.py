import json

from annotuneagent.decide import (
    SYSTEM_ML,
    SYSTEM_NAME,
    cluster_payload,
    ml_stat_row,
    ml_user,
    parse_decision,
    parse_ml,
    payload_agreement,
    propose_hypotheses,
    trial_support,
    wilcoxon_payload,
)
from annotuneagent.hypotheses import Hypothesis


def test_ml_user_keeps_negative_marker_measurements():
    from annotuneagent.decide import ml_user_from_hits

    payload = {
        "species": "human",
        "tissue": "blood",
        "stage": "",
        "dictionary_hits": [{"name": "T", "genes": ["CD3E"]}],
    }
    hypothesis = Hypothesis("T", positive=("CD3E",), negative=("MS4A1",))
    measurements = {
        "CD3E": {"gene": "CD3E", "logfc": 2.0, "pct_in": 0.9, "pct_out": 0.1},
        "MS4A1": {"gene": "MS4A1", "logfc": -1.0, "pct_in": 0.05, "pct_out": 0.4},
    }
    user = ml_user_from_hits(payload, [hypothesis], measurements, [measurements["CD3E"]])
    rows = {row[0]: row for row in user["marker_measurements"]}
    assert rows["MS4A1"] == ["MS4A1", -1.0, 0.05, 0.4]


def test_hit_order_ignores_the_file_name():
    from annotuneagent.decide import ranked_payload

    order = [("A", 4.0), ("B", 3.0), ("C", 2.0), ("D", 1.0)]
    pct = {gene: 0.9 for gene in "ABCD"}
    hypotheses = [Hypothesis(gene, positive=(gene,)) for gene in "ABCD"]
    renamed = ranked_payload(order, pct, hypotheses, source="run1.h5ad", donor="d", cluster="0")
    copied = ranked_payload(
        order, pct, hypotheses, source="copy_of_run1.h5ad", donor="d", cluster="0"
    )
    api = ranked_payload(order, pct, hypotheses, donor="d", cluster="0")
    names = [item["name"] for item in renamed["dictionary_hits"]]
    assert names == [item["name"] for item in copied["dictionary_hits"]]
    assert names == [item["name"] for item in api["dictionary_hits"]]
    assert names != ["A", "B", "C", "D"]


def test_unmeasured_negative_is_a_null_measurement():
    genes = [{"gene": "CD3E", "logfc": 2.0, "pct_in": 0.9, "pct_out": 0.1}]
    user = ml_user(
        "human",
        "blood",
        "",
        genes,
        [("T cell", ["CD3E"], ["MS4A1", "TRDC"])],
        {"CD3E": {"gene": "CD3E", "logfc": 2.0, "pct_in": 0.9, "pct_out": 0.1}},
    )
    assert user["candidates"][0][3][6] == 2
    rows = {row[0]: row for row in user["marker_measurements"]}
    assert rows["MS4A1"] == ["MS4A1", None, None, None]
    assert rows["TRDC"] == ["TRDC", None, None, None]


def test_undetected_gene_is_not_a_dictionary_hit():
    from annotuneagent.decide import ranked_payload

    payload = ranked_payload(
        [("Nkg7", 2.5), ("Prph", 1.5), ("Uchl1", 2.6)],
        {"Nkg7": 0.003, "Prph": 0.42, "Uchl1": 0.94},
        [
            Hypothesis("NK cell", positive=("Nkg7",)),
            Hypothesis("sensory neuron", positive=("Prph",)),
        ],
        species="mouse",
        tissue="drg",
    )
    assert [item["name"] for item in payload["dictionary_hits"]] == ["sensory neuron"]
    assert "Nkg7" not in payload["dictionary_hits"][0]["genes"]


def test_t_subtype_without_a_receptor_is_not_a_hit():
    from annotuneagent.decide import ranked_payload

    payload = ranked_payload(
        [("S100a4", 2.0), ("Cd44", 1.5), ("Lyz2", 3.0)],
        {"S100a4": 0.8, "Cd44": 0.7, "Lyz2": 0.9, "Cd3e": 0.02},
        [
            Hypothesis(
                "memory CD4 T cell",
                positive=("S100a4", "Cd44"),
                requires_any=("Cd3e", "Cd3d"),
            ),
            Hypothesis("monocyte", positive=("Lyz2",)),
        ],
    )
    assert [item["name"] for item in payload["dictionary_hits"]] == ["monocyte"]


def test_name_outside_the_hit_list_abstains():
    text = '{"action": "retest", "names": ["NK cell"]}'
    assert parse_decision(text, {"sensory neuron"}).action == "escalate"
    assert parse_ml('{"names": ["NK cell"]}', {"sensory neuron"}).status == "escalate"


def test_parse_decision_keeps_only_allowed_names():
    text = json.dumps(
        {
            "action": "retest",
            "names": ["NK cell", "made up"],
            "markers": [{"name": "special", "positive": ["X"], "negative": []}],
        }
    )
    decision = parse_decision(text, {"NK cell"})
    assert decision.names == ("NK cell",)
    assert decision.markers[0].name == "special"
    assert parse_decision('{"action": "refuse"}').action == "refuse"
    escalated = parse_decision('{"action": "escalate", "names": ["NK cell"]}', {"NK cell"})
    assert escalated.action == "escalate"
    assert escalated.names == ()


def test_propose_uses_enriched_genes():
    catalog = [
        Hypothesis("B cell", positive=("MS4A1",)),
        Hypothesis("T cell", positive=("CD3E",)),
    ]
    proposed = propose_hypotheses(catalog, ["MS4A1"], {"T cell"})
    assert [item.name for item in proposed] == ["B cell"]


def test_parse_ml_accepts_names_and_escalate():
    named = parse_ml('{"names":["hepatocyte"]}', {"hepatocyte"})
    assert named.status == "named"
    assert named.names == ("hepatocyte",)
    assert parse_ml('{"action":"escalate"}').status == "escalate"
    try:
        parse_decision('{"names":["hepatocyte"]}')
    except ValueError as exc:
        assert "unknown action" in str(exc)
    else:
        raise AssertionError("name parser accepted an ml success")
    positive = [{"gene": "ALB", "logfc": 25.0, "pct_in": 0.01, "pct_out": 0.0}]
    stats = ml_stat_row(positive, [], {row["gene"]: row for row in positive})
    assert stats[1] == 1
    trial = trial_support(positive, ["PTPRC"], {})
    assert trial["n_positive_supported_detected"] == 0
    user = ml_user(
        "human", "liver", "", positive, [("hepatocyte", ["ALB"], [])], {"ALB": positive[0]}
    )
    assert "ml_trial" not in user
    assert user["candidate_stat_columns"][-1] == "mean_positive_logfc"
    assert "names" in SYSTEM_ML


def test_cluster_payload_lists_genes_and_dictionary_hits():
    import numpy as np
    from anndata import AnnData

    from annotuneagent.hypotheses import Hypothesis

    x = np.zeros((40, 3), dtype=float)
    x[:20, 0] = 5
    x[:20, 1] = 4
    x[20:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["CD3E", "CD4", "CD8A"]
    adata.uns["log1p"] = {"base": None}
    payload = cluster_payload(
        adata,
        "leiden",
        "0",
        [Hypothesis("CD4 T", positive=("CD3E", "CD4"), negative=("CD8A",))],
        species="human",
        tissue="blood",
    )
    assert set(payload) == {"species", "tissue", "stage", "genes", "dictionary_hits"}
    listed = {item["gene"] for item in payload["genes"]}
    assert payload["dictionary_hits"][0]["name"] == "CD4 T"
    assert set(payload["dictionary_hits"][0]["genes"]) <= listed
    wilcox = wilcoxon_payload(
        adata,
        "leiden",
        "0",
        [Hypothesis("CD4 T", positive=("CD3E", "CD4"), negative=("CD8A",))],
        species="human",
        tissue="blood",
    )
    assert set(wilcox) == set(payload)
    assert payload_agreement(payload, wilcox)["same_top25_order"] is True
    assert "CD4 T" in {item["name"] for item in payload["dictionary_hits"]}
    assert SYSTEM_NAME != SYSTEM_ML


def test_marker_entry_without_a_name_is_a_value_error():
    import pytest

    from annotuneagent.decide import parse_decision

    reply = json.dumps({"action": "retest", "markers": [{"positive": ["CD3E"]}]})
    with pytest.raises(ValueError, match="needs a name"):
        parse_decision(reply)
    with pytest.raises(ValueError, match="must be a list"):
        parse_decision(
            json.dumps({"action": "retest", "markers": [{"name": "T", "positive": "CD3E"}]})
        )


def test_no_hits_means_no_name_is_allowed():
    from annotuneagent.decide import parse_decision, parse_ml

    named = json.dumps({"action": "retest", "names": ["invented cell"]})
    assert parse_decision(named, set()).action == "escalate"
    assert parse_ml(json.dumps({"names": ["invented cell"]}), set()).status == "escalate"


def test_chat_reply_without_choices_is_a_runtime_error(monkeypatch):
    import io

    import pytest

    from annotuneagent import decide

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        decide.urllib.request,
        "urlopen",
        lambda *_a, **_k: Reply(b'{"error": {"message": "slot busy"}}'),
    )
    with pytest.raises(RuntimeError, match="no message content"):
        decide.post_chat("http://x", "system", {"a": 1})


def test_fetch_name_reads_a_completion(monkeypatch):
    from annotuneagent import decide

    monkeypatch.setattr(
        "annotuneagent.local_lora.complete_url",
        lambda url, task, system, payload, *, maxlen, timeout, tokenizer: json.dumps(
            {"action": "retest", "names": ["T cell"]}
        ),
    )
    decision = decide.fetch_name("http://127.0.0.1:8080", {"dictionary_hits": [{"name": "T cell"}]})
    assert decision.action == "retest"
    assert decision.names == ("T cell",)


def test_rank_measurements_reads_detection_for_ranked_genes():
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    from annotuneagent.decide import rank_measurements

    names = {"0": ["G1", "G2", "G3"]}
    adata = AnnData(X=np.zeros((2, 3)))
    adata.uns["rank_genes_groups"] = {
        "names": names,
        "logfoldchanges": {"0": [2.0, -1.0, 0.5]},
        "pts": pd.DataFrame({"0": [0.9, 0.5, 0.1]}, index=["G1", "G2", "G3"]),
        "pts_rest": pd.DataFrame({"0": [0.2, 0.4, 0.0]}, index=["G3", "G2", "G1"]),
    }
    positive, measured = rank_measurements(adata, "0")
    assert [row["gene"] for row in positive] == ["G1", "G3"]
    assert measured["G1"] == {"gene": "G1", "logfc": 2.0, "pct_in": 0.9, "pct_out": 0.0}
    assert measured["G3"]["pct_out"] == 0.2
