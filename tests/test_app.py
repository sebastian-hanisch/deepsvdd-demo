"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, Abspielen, ausgeblendete Regler, Experimente auf Abruf, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dsvdd_constants as C
from dsvdd_evaluation import SWEEP_LABELS
from dsvdd_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
EXPECTED_KIND = {"Standardfall (Voreinstellung)": None, "Ohne Training (0 Epochen)": "success", "ReLU, 1000 Epochen": "warning", "Zentrierte Eingaben": "warning", "Gekrümmter Normalbereich": None,
                 "Korrelationsbruch": "warning", "Viele Anomalien (45 %)": "warning"}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    """Erst die Betriebsarten setzen und laufen lassen (die Art "Lücke" hängt davon ab; AppTest prüft gegen die Optionen des vorigen Laufs), dann alles andere."""
    at.session_state["n_modes_slider"] = p["n_modes"]
    at.run()
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox)}


def _slider(at, prefix):
    return [s for s in at.sidebar.slider if s.label.startswith(prefix)][0]


def test_default_renders_without_exception():
    at = _run()
    assert any("Deep SVDD in Aktion" in m.value for m in at.markdown)
    assert not at.error and len(at.warning) + len(at.success) == 1


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    kind = EXPECTED_KIND[name]
    if kind is None:
        assert len(at.success) + len(at.warning) == 1
    else:
        assert (len(at.success) == 1 and not at.warning) if kind == "success" else (len(at.warning) == 1 and not at.success)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["p_slider"] = C.P_MAX
        at.session_state["n_noise_slider"] = C.N_NOISE_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MIN
        at.session_state["epochs_slider"] = 0
        at.session_state["width_select"] = 4
        at.session_state["depth_slider"] = 0
        at.session_state["n_out_slider"] = 1
    _run(small)

    def large(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MAX
        at.session_state["p_slider"] = C.P_MIN
        at.session_state["n_modes_slider"] = C.N_MODES_MAX
        at.session_state["curvature_slider"] = C.CURVATURE_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MAX
        at.session_state["kind_select"] = "cluster"
        at.session_state["strength_slider"] = C.STRENGTH_MIN
        at.session_state["epochs_slider"] = C.EPOCHS_MAX
        at.session_state["width_select"] = 64
        at.session_state["depth_slider"] = C.DEPTH_MAX
        at.session_state["n_out_slider"] = C.N_OUT_MAX
        at.session_state["bias_toggle"] = True
        at.session_state["activation_select"] = "relu"
    _run(large)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 12, 30])
def test_every_step_renders(step, p):
    def setup(at):
        at.session_state["p_slider"] = p
        at.session_state["dsvdd_step"] = step
    _run(setup)


@pytest.mark.parametrize("variant", ["one_class", "soft_boundary"])
@pytest.mark.parametrize("kind", ["scattered", "decorrelated", "cluster"])
def test_every_step_renders_for_both_variants_and_every_kind(variant, kind):
    def setup(at):
        at.session_state["variant_select"] = variant
        at.session_state["kind_select"] = kind
        at.session_state["epochs_slider"] = 0 if kind == "cluster" else 30
        at.session_state["bias_toggle"] = kind == "decorrelated"
    for step in (1, 2, 3, 4, 5):
        at = _run(setup)
        at.session_state["dsvdd_step"] = step
        at.run()
        assert not at.exception, [e.value for e in at.exception]


def test_every_step_renders_with_few_tours_many_features_noise_and_the_share_threshold():
    def setup(at):
        at.session_state["n_tours_slider"] = 20
        at.session_state["p_slider"] = 30
        at.session_state["n_noise_slider"] = 40
        at.session_state["epochs_slider"] = 20
    for step in (1, 2, 3, 4, 5):
        at = _run(setup)
        at.session_state["dsvdd_step"] = step
        at.run()
        assert not at.exception
    at = _run(lambda a: a.session_state.__setitem__("threshold_kind_select", "share"))
    for step in (3, 4, 5):
        at.session_state["dsvdd_step"] = step
        at.run()
        assert not at.exception


def test_the_play_button_walks_through_all_steps_without_duplicate_keys():
    at = _run()
    [b for b in at.button if b.label.startswith("▶️")][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) >= 2


def test_threshold_controls_are_hidden_per_kind_and_their_values_are_kept():
    at = _run(lambda a: a.session_state.__setitem__("quantile_slider", 0.99))
    assert {"Schwelle: χ²-Quantil (robust)"} <= _labels(at) and "Angenommener Anteil der Anomalien [%]" not in _labels(at)
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    assert not at.exception and "Angenommener Anteil der Anomalien [%]" in _labels(at) and "Schwelle: χ²-Quantil (robust)" not in _labels(at)
    at.session_state["share_slider"] = 25
    at.run()
    at.session_state["threshold_kind_select"] = "standard"
    at.run()
    assert not at.exception
    assert _slider(at, "Schwelle: χ²").value == 0.99
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    assert _slider(at, "Angenommener").value == 25


def test_sweep_options_follow_the_threshold_kind_and_the_hidden_kinds():
    at = _run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert SWEEP_LABELS["nu"] in options and SWEEP_LABELS["epochs"] in options and SWEEP_LABELS["quantile"] in options and SWEEP_LABELS["share"] not in options
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert SWEEP_LABELS["share"] in options and SWEEP_LABELS["quantile"] not in options and SWEEP_LABELS["nu"] in options
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert not ({SWEEP_LABELS["strength"], SWEEP_LABELS["n_modes"]} & set(options)) and SWEEP_LABELS["contamination"] in options
    at.session_state["kind_select"] = "decorrelated"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert SWEEP_LABELS["strength"] not in options and SWEEP_LABELS["n_modes"] in options


def test_strength_slider_is_hidden_for_gap_and_correlation_break_and_its_value_is_kept():
    at = _run(lambda a: (a.session_state.__setitem__("strength_slider", 9.0)))
    assert "Abstand der Anomalien (Faktor-σ)" in _labels(at)
    at.session_state["kind_select"] = "decorrelated"
    at.run()
    assert not at.exception and "Abstand der Anomalien (Faktor-σ)" not in _labels(at)
    at.session_state["kind_select"] = "scattered"
    at.run()
    assert not at.exception and _slider(at, "Abstand").value == 9.0
    at.session_state["n_modes_slider"] = 2
    at.run()
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and "Abstand der Anomalien (Faktor-σ)" not in _labels(at)
    at.session_state["kind_select"] = "scattered"
    at.run()
    assert _slider(at, "Abstand").value == 9.0


def test_gap_kind_needs_two_modes_and_falls_back_when_modes_drop():
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    at.session_state["n_modes_slider"] = 1
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] not in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    assert at.session_state["kind_select"] == "scattered"
    assert C.KIND_LABELS["decorrelated"] in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["dsvdd_step"] = 4
    at.run()
    assert not at.exception and at.session_state["dsvdd_step"] == 4
    at.session_state["contamination_slider"] = 20
    at.run()
    assert not at.exception and at.session_state["dsvdd_step"] == 1


def test_experiments_run_on_demand_and_sweeps_run():
    at = _run()
    for parameter in ("epochs", "width", "nu", "n_noise", "contamination"):
        [s for s in at.selectbox if s.key == "sweep_select"][0].select(parameter)
        at.run()
        [b for b in at.button if b.key == "sweep_start"][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    keys = ("curves", "seeds", "collapse", "width", "scenarios", "threshold", "cost")
    for key in keys:
        [b for b in at.button if b.key == f"{key}_start"][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert all(at.session_state[f"{k}_on"] for k in keys)


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "dsvdd_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M))
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_unique_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    parts = source.split("plotly_chart(")[1:]
    keys = [re.search(r'key="([a-z0-9_]+)"', part.split("plotly_chart(")[0]) for part in parts]
    assert len(parts) >= 18 and all(keys)
    names = [k.group(1) for k in keys]
    assert len(set(names)) == len(names)


def test_app_has_no_dead_file_links_and_the_verbatim_footer():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.findall(r"\]\([a-z_]+\.py\)", source)
    assert "https://sebastianhanisch.net/kontakt.html" in source and "Interesse an einer maßgeschneiderten Lösung für" in source


def test_net_seed_button_changes_only_the_network_seed_and_the_bias_toggle_is_a_real_checkbox():
    at = _run()
    assert at.session_state["net_seed_input"] == C.DEFAULT_NET_SEED and at.session_state["seed_input"] == C.DEFAULT_SEED
    [b for b in at.sidebar.button if b.label.startswith("🎲 Neues Netz")][0].click()
    at.run()
    assert not at.exception and at.session_state["net_seed_input"] != C.DEFAULT_NET_SEED and at.session_state["seed_input"] == C.DEFAULT_SEED
    box = [c for c in at.sidebar.checkbox if c.key == "bias_toggle"][0]
    assert box.value is False
    box.check()
    at.run()
    assert not at.exception and at.session_state["bias_toggle"] is True


def test_zero_epochs_shows_the_untrained_network_and_the_verdict_is_not_training_hurts():
    at = _run(lambda a: a.session_state.__setitem__("epochs_slider", 0))
    assert not any("Das Training schadet" in w.value for w in at.warning)
