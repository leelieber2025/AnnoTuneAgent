from annotuneagent.hypotheses import Hypothesis
from annotuneagent.votes import agreed_name, gene_vote, panel_vote, subcluster_vote


def test_three_views_write_the_name_and_one_dissent_does_not_block():
    votes = {
        "panel": "oligodendrocyte",
        "genes": "oligodendrocyte",
        "subcluster": "oligodendrocyte",
        "neighbors": "astrocyte",
    }
    assert agreed_name(votes, {}) == "oligodendrocyte"


def test_two_coarse_names_stay_unresolved():
    votes = {"panel": "T cell", "genes": "T cell", "subcluster": "B cell", "neighbors": "B cell"}
    assert agreed_name(votes, {}) is None


def test_parent_is_written_when_children_split_the_votes():
    parents = {"oligodendrocyte precursor cell": "oligodendrocyte"}
    votes = {
        "panel": "oligodendrocyte",
        "genes": "oligodendrocyte",
        "subcluster": "oligodendrocyte precursor cell",
        "neighbors": "oligodendrocyte precursor cell",
    }
    assert agreed_name(votes, parents) == "oligodendrocyte"


def test_three_child_votes_keep_the_child():
    parents = {"oligodendrocyte precursor cell": "oligodendrocyte"}
    votes = {
        "panel": "oligodendrocyte precursor cell",
        "genes": "oligodendrocyte precursor cell",
        "subcluster": "oligodendrocyte precursor cell",
        "neighbors": "oligodendrocyte",
    }
    assert agreed_name(votes, parents) == "oligodendrocyte precursor cell"


def test_two_views_are_not_enough():
    assert (
        agreed_name(
            {"panel": "T cell", "genes": "T cell", "subcluster": None, "neighbors": None}, {}
        )
        is None
    )


def test_gene_vote_ignores_a_single_high_marker_when_another_was_measured():
    hypotheses = [Hypothesis("beta", positive=("INS", "IAPP"), negative=())]
    assert gene_vote(hypotheses, {"INS": 4.0, "IAPP": -0.2}) is None
    assert gene_vote(hypotheses, {"INS": 4.0}) == "beta"


def test_panel_vote_skips_a_high_negative_marker():
    record = {
        "status": "uncertain",
        "ranking": [{"name": "beta", "score": 1.2}],
        "evidence": {"negative": [{"gene": "GCG", "pct_in": 0.4, "logfc": 2.0}]},
    }
    assert panel_vote(record) is None


def test_subcluster_vote_requires_one_shared_name():
    assert subcluster_vote([{"status": "label", "cell_type": "T cell"}]) == "T cell"
    assert (
        subcluster_vote(
            [
                {"status": "label", "cell_type": "T cell"},
                {"status": "label", "cell_type": "B cell"},
            ]
        )
        is None
    )
