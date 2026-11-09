import numpy as np
import pytest
from anndata import AnnData

from annotuneagent.evidence import marker_statistics


def test_marker_statistics_names_the_missing_gene():
    adata = AnnData(X=np.ones((6, 2)))
    adata.var_names = ["A", "B"]
    mask = np.array([True] * 3 + [False] * 3)
    with pytest.raises(KeyError, match="'C'"):
        marker_statistics(adata, [mask], ["A", "C"])


def test_marker_statistics_rejects_duplicate_var_names():
    adata = AnnData(X=np.ones((6, 2)))
    adata.var_names = ["A", "A"]
    mask = np.array([True] * 3 + [False] * 3)
    with pytest.raises(ValueError, match="not unique"):
        marker_statistics(adata, [mask], ["A"])
