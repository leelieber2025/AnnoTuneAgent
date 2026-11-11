import csv
from pathlib import Path

import pytest

from annotuneagent.markers import hypotheses_from_names, marker_names, suggest_tissues, tissue_pages


def test_core_hides_tissue_rows():
    names = marker_names("human")
    assert "T cell" in names
    assert "pulmonary alveolar type 2 cell" not in names
    assert "Kupffer cell" not in marker_names("mouse")


def test_tissue_adds_only_that_page():
    lung = marker_names("human", "lung")
    assert "pulmonary alveolar type 2 cell" in lung
    assert "T cell" in lung
    assert "hepatocyte" not in lung


def test_cd4_is_support_and_platelet_rejects_lysozyme():
    human = hypotheses_from_names(["CD4-positive, alpha-beta T cell"], "human")[0]
    mouse = hypotheses_from_names(["CD4-positive, alpha-beta T cell"], "mouse")[0]
    assert human.positive == ("CD3E", "IL7R")
    assert human.support == ("CD4",)
    assert "CD8A" in human.negative
    assert mouse.positive == ("Cd3e", "Il7r")
    assert mouse.support == ("Cd4",)
    assert "LYZ" in hypotheses_from_names(["platelet"], "human")[0].negative
    assert "Lyz2" in hypotheses_from_names(["platelet"], "mouse")[0].negative


def test_mouse_symbols_are_not_human_copies():
    nk = hypotheses_from_names(["NK cell"], "mouse")[0]
    assert "GNLY" not in nk.positive
    assert "Nkg7" in nk.positive
    monocyte = hypotheses_from_names(["monocyte"], "mouse")[0]
    assert "Lyz2" in monocyte.positive
    chief = hypotheses_from_names(["gastric chief cell"], "mouse", "stomach")[0]
    assert "Cblif" in chief.positive
    assert "Gif" not in chief.positive
    rpe = hypotheses_from_names(["retinal pigment epithelial cell"], "mouse", "retina")[0]
    assert rpe.positive == ("Rpe65", "Lrat")
    assert "Best1" not in rpe.positive
    c_cell = hypotheses_from_names(["parafollicular cell"], "mouse", "thyroid")[0]
    assert c_cell.positive == ("Calca",)
    assert c_cell.support == ("Chga",)


def test_tissue_type_requires_tissue():
    with pytest.raises(KeyError, match="lung"):
        hypotheses_from_names(["pulmonary alveolar type 2 cell"], "human")


def test_scoring_table_drops_conflicts():
    from annotuneagent.markers import load_entries

    entries = load_entries()
    by = {(e.species, e.cell_type): e for e in entries}
    assert ("human", "NKT cell") not in by
    assert ("human", "dendritic cell") not in by
    assert ("human", "epithelial cell") not in by
    memory = by[("human", "memory CD4 T cell")]
    assert "CCR7" not in memory.negative
    assert "SLC4A10" in memory.negative
    assert "TRAV1-2" in memory.negative
    ilc2 = by[("human", "ILC2")]
    assert "HPGDS" in ilc2.positive
    assert "IL1RL1" not in ilc2.positive
    assert "IL33R" not in ilc2.positive
    assert by[("human", "plasma cell")].parent is None
    assert "SLC17A6" not in by[("human", "excitatory neuron")].positive
    assert ("human", "lung goblet cell") in by
    assert ("human", "intestine goblet cell") in by
    assert ("human", "goblet cell") not in by


def test_t_cell_descendants_require_a_receptor_gene():
    mouse_memory = hypotheses_from_names(["memory CD4 T cell"], "mouse")[0]
    human_memory = hypotheses_from_names(["memory CD4 T cell"], "human")[0]
    mouse_t = hypotheses_from_names(["T cell"], "mouse")[0]
    mouse_nk = hypotheses_from_names(["NK cell"], "mouse")[0]
    assert mouse_memory.requires_any == ("Cd3d", "Cd3e", "Cd3g", "Trac")
    assert human_memory.requires_any == ("CD3D", "CD3E", "CD3G", "TRAC")
    assert mouse_t.requires_any == ()
    assert mouse_nk.requires_any == ()


def test_species_does_not_cross():
    human = hypotheses_from_names(["monocyte"], "human")[0]
    mouse = hypotheses_from_names(["monocyte"], "mouse")[0]
    assert "CD14" in human.positive
    assert "Lyz2" in mouse.positive
    assert "CD14" not in mouse.positive


def test_dictionary_has_one_candidate_per_monocyte_subset():
    assert "classical monocyte" not in marker_names("human")
    assert "non-classical monocyte" not in marker_names("human")
    assert "CD14-positive monocyte" not in marker_names("mouse")
    assert "CD16-positive monocyte" not in marker_names("mouse")


def test_mouse_pan_ilc_is_brain_scoped_and_subtypes_stay():
    assert "innate lymphoid cell" not in marker_names("mouse")
    assert "innate lymphoid cell" in marker_names("mouse", "brain")
    assert {"ILC2", "ILC3"} <= set(marker_names("mouse"))


def test_author_drg_offers_the_drg_page():
    assert suggest_tissues("DRG", "mouse") == ["drg"]
    assert suggest_tissues("dorsal root ganglion") == ["drg"]
    assert "brain" not in suggest_tissues("DRG")


@pytest.mark.parametrize(
    "author, species, page",
    [
        ("corneal epithelium", "human", "cornea"),
        ("retinal tissue", "mouse", "retina"),
        ("placental villi", "human", "placenta"),
        ("endometrial biopsy", "human", "endometrium"),
        ("ovarian cortex", "mouse", "ovary"),
        ("testicular biopsy", "human", "testis"),
        ("breast tissue", "human", "mammary"),
        ("thymic stroma", "human", "thymus"),
    ],
)
def test_common_author_tissue_aliases(author, species, page):
    assert suggest_tissues(author, species) == [page]


def test_unknown_author_tissue_returns_every_page():
    assert suggest_tissues("osteomyelitis lesion", "mouse") == tissue_pages("mouse")
    assert "drg" in tissue_pages("mouse")
    assert "brain" in tissue_pages("mouse")


@pytest.mark.parametrize(
    "scope, cell_type",
    [
        ("adipose", "adipocyte"),
        ("muscle", "skeletal muscle satellite cell"),
        ("muscle", "skeletal muscle fiber"),
        ("bone", "osteoblast"),
        ("bone", "osteoclast"),
        ("heart", "cardiac fibroblast"),
        ("tendon", "tenocyte"),
        ("stomach", "foveolar cell"),
        ("stomach", "mucous neck cell"),
        ("stomach", "parietal cell"),
        ("stomach", "gastric chief cell"),
        ("cartilage", "chondrocyte"),
        ("cartilage", "superficial zone articular chondrocyte"),
        ("cartilage", "hypertrophic chondrocyte"),
        ("drg", "satellite glial cell"),
        ("drg", "myelinating Schwann cell"),
        ("drg", "non-myelinating Schwann cell"),
        ("drg", "sensory neuron"),
        ("drg", "peptidergic nociceptor"),
        ("drg", "non-peptidergic nociceptor"),
        ("drg", "large diameter sensory neuron"),
        ("drg", "proprioceptive neuron"),
        ("drg", "C-LTMR"),
        ("bone", "osteocyte"),
        ("adipose", "brown adipocyte"),
        ("adipose", "preadipocyte"),
        ("heart", "endocardial cell"),
        ("skin", "sebocyte"),
        ("stomach", "enterochromaffin-like cell"),
        ("pancreas", "pancreatic stellate cell"),
        ("thyroid", "thyroid follicular cell"),
        ("thyroid", "parafollicular cell"),
        ("bladder", "bladder urothelial cell"),
        ("salivary", "serous acinar cell of salivary gland"),
        ("salivary", "mucous acinar cell of salivary gland"),
        ("salivary", "basal duct cell of salivary gland"),
        ("salivary", "myoepithelial cell of salivary gland"),
        ("retina", "retinal rod cell"),
        ("retina", "retinal cone cell"),
        ("retina", "retinal bipolar neuron"),
        ("retina", "retinal ganglion cell"),
        ("retina", "retinal pigment epithelial cell"),
        ("testis", "Sertoli cell"),
        ("testis", "Leydig cell"),
        ("testis", "male germ cell"),
        ("ovary", "granulosa cell"),
        ("ovary", "theca cell"),
        ("ovary", "oocyte"),
        ("mammary", "basal-myoepithelial cell of mammary gland"),
        ("mammary", "luminal epithelial cell of mammary gland"),
        ("prostate", "luminal cell of prostate epithelium"),
        ("prostate", "basal cell of prostate epithelium"),
        ("endometrium", "glandular epithelial cell of endometrium"),
        ("endometrium", "stromal cell of endometrium"),
        ("retina", "Mueller cell"),
        ("cornea", "corneal epithelial cell"),
        ("cornea", "corneal endothelial cell"),
        ("cornea", "keratocyte"),
        ("adrenal", "cortical cell of adrenal gland"),
        ("adrenal", "adrenal medulla chromaffin cell"),
        ("thymus", "epithelial cell of thymus"),
        ("intestine", "enteric neuron"),
        ("heart", "mesothelial cell of epicardium"),
        ("brain", "choroid plexus epithelial cell"),
        ("kidney", "kidney connecting tubule epithelial cell"),
        ("retina", "retina horizontal cell"),
        ("retina", "amacrine cell"),
        ("muscle", "skeletal muscle myoblast"),
        ("ovary", "stromal cell of ovary"),
        ("skin", "Merkel cell"),
        ("intestine", "transit amplifying cell of small intestine"),
        ("limb", "Schwann cell"),
    ],
)
@pytest.mark.parametrize("species", ["human", "mouse"])
def test_new_tissue_types_are_reachable_only_in_their_tissue(species, scope, cell_type):
    assert cell_type not in marker_names(species)
    assert cell_type in marker_names(species, scope)
    assert hypotheses_from_names([cell_type], species, scope)[0].positive


def test_eosinophil_uses_species_specific_positive_markers():
    human = hypotheses_from_names(["eosinophil"], "human")[0]
    mouse = hypotheses_from_names(["eosinophil"], "mouse")[0]
    assert human.positive == ("CLC", "SIGLEC8")
    assert mouse.positive == ("Epx", "Prg2")


@pytest.mark.parametrize(
    "cell_type",
    [
        "mononuclear cytotrophoblast cell",
        "syncytiotrophoblast cell",
        "extravillous trophoblast",
    ],
)
def test_human_placental_types_have_no_mouse_copy(cell_type):
    assert cell_type in marker_names("human", "placenta")
    assert cell_type not in marker_names("mouse")
    assert "placenta" not in tissue_pages("mouse")
    with pytest.raises(ValueError, match="unknown mouse tissue"):
        marker_names("mouse", "placenta")


@pytest.mark.parametrize("species", ["human", "mouse"])
@pytest.mark.parametrize(
    "scope, cell_type, positive",
    [
        ("stomach", "type G cell of stomach", ("GAST", "Gast")),
        ("stomach", "type D cell of stomach", ("SST", "Sst")),
        ("stomach", "tuft cell of stomach", ("POU2F3", "Pou2f3")),
        ("intestine", "M cell of gut", ("GP2", "Gp2")),
        ("heart", "regular atrial cardiac myocyte", ("MYL7", "Myl7")),
        ("heart", "regular ventricular cardiac myocyte", ("MYL3", "Myl2")),
        ("kidney", "mesangial cell", ("ITGA8", "Itga8")),
        ("kidney", "glomerular capillary endothelial cell", ("EHD3", "Ehd3")),
        ("kidney", "macula densa epithelial cell", ("NOS1", "Nos1")),
        ("pancreas", "pancreatic epsilon cell", ("GHRL", "Ghrl")),
        ("intestine", "interstitial cell of Cajal", ("ANO1", "Ano1")),
        ("kidney", "parietal epithelial cell", ("CLDN1", "Cldn1")),
    ],
)
def test_literature_marker_additions_are_tissue_scoped(species, scope, cell_type, positive):
    assert cell_type not in marker_names(species)
    marker = hypotheses_from_names([cell_type], species, scope)[0]
    assert positive[species == "mouse"] in marker.positive


def test_marker_rows_and_ontology_map_use_the_same_cl_ids():
    from annotuneagent.granularity.ontology_map import load_map

    root = Path(__file__).resolve().parents[1]
    label_map = load_map(root / "src" / "annotuneagent" / "ontology" / "ontology_map.jsonl")
    with (root / "src" / "annotuneagent" / "markers.tsv").open(newline="") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            assert "lit=PMID:" in entry["evidence"], (entry["species"], entry["cell_type"])
            row = label_map.lookup(entry["cell_type"], entry["species"])
            assert row is not None, (entry["species"], entry["cell_type"])
            assert row.cl_id == entry["cl_id"], (entry["species"], entry["cell_type"])


def test_every_parent_is_a_row_of_the_same_species():
    from annotuneagent.markers import load_entries

    rows = load_entries()
    known = {(entry.species, entry.cell_type) for entry in rows}
    assert all((e.species, e.parent) in known for e in rows if e.parent is not None)
    removed = {"epithelial cell", "dendritic cell", "macrophage", "keratinocyte"}
    assert not removed & {entry.parent for entry in rows}


def test_a_parent_that_is_not_a_row_is_rejected(tmp_path, monkeypatch):
    import annotuneagent.markers as markers

    header = "species\tscope\tcell_type\tparent\tpositive\tnegative\n"
    table = tmp_path / "markers.tsv"
    table.write_text(header + "human\tany\tenterocyte\tepithelial cell\tFABP2\t\n")
    monkeypatch.setattr(markers, "_TABLE", table)
    with pytest.raises(ValueError, match="parent 'epithelial cell' is not a human row"):
        markers.load_entries()
