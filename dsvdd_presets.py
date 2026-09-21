"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio, siehe ocsvm_presets.py in ocsvm-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dsvdd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _width(value):
    """Ein beliebiger Zahlenwert wird auf die nächste Stufe der Breite gelegt."""
    v = float(value)
    return min(C.WIDTH_OPTIONS, key=lambda w: abs(w - v))


def _bool(value):
    value = str(value).lower()
    if value in ("1", "true", "ja"):
        return True
    if value in ("0", "false", "nein"):
        return False
    raise ValueError(value)


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "p_slider": SettingSpec("p", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "n_noise_slider": SettingSpec("nn", int, C.DEFAULT_N_NOISE, C.N_NOISE_MIN, C.N_NOISE_MAX),
    "n_modes_slider": SettingSpec("modes", int, C.DEFAULT_N_MODES, C.N_MODES_MIN, C.N_MODES_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "contamination_slider": SettingSpec("cont", int, C.DEFAULT_CONTAMINATION, C.CONTAMINATION_MIN, C.CONTAMINATION_MAX),
    "kind_select": SettingSpec("kind", _choice(C.KINDS), C.DEFAULT_KIND),
    "strength_slider": SettingSpec("str", float, C.DEFAULT_STRENGTH, C.STRENGTH_MIN, C.STRENGTH_MAX),
    "variant_select": SettingSpec("var", _choice(C.VARIANTS), C.DEFAULT_VARIANT),
    "width_select": SettingSpec("w", _width, C.DEFAULT_WIDTH, min(C.WIDTH_OPTIONS), max(C.WIDTH_OPTIONS)),
    "depth_slider": SettingSpec("d", int, C.DEFAULT_DEPTH, C.DEPTH_MIN, C.DEPTH_MAX),
    "n_out_slider": SettingSpec("out", int, C.DEFAULT_N_OUT, C.N_OUT_MIN, C.N_OUT_MAX),
    "epochs_slider": SettingSpec("ep", int, C.DEFAULT_EPOCHS, C.EPOCHS_MIN, C.EPOCHS_MAX),
    "nu_slider": SettingSpec("nu", float, C.DEFAULT_NU, C.NU_MIN, C.NU_MAX),
    "activation_select": SettingSpec("act", _choice(C.ACTIVATIONS), C.DEFAULT_ACTIVATION),
    "bias_toggle": SettingSpec("bias", _bool, C.DEFAULT_BIAS),
    "scaling_select": SettingSpec("sc", _choice(C.SCALINGS), C.DEFAULT_SCALING),
    "net_seed_input": SettingSpec("ns", int, C.DEFAULT_NET_SEED, 0, 1_000_000),
    "threshold_kind_select": SettingSpec("thr", _choice(C.THRESHOLD_KINDS), C.DEFAULT_THRESHOLD_KIND),
    "quantile_slider": SettingSpec("q", float, C.DEFAULT_QUANTILE, C.QUANTILE_MIN, C.QUANTILE_MAX),
    "share_slider": SettingSpec("share", int, C.DEFAULT_SHARE, C.SHARE_MIN, C.SHARE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"n": "n_tours_slider", "p": "p_slider", "n_noise": "n_noise_slider", "n_modes": "n_modes_slider", "curvature": "curvature_slider", "noise": "noise_slider",
               "contamination": "contamination_slider", "kind": "kind_select", "strength": "strength_slider", "variant": "variant_select", "width": "width_select", "depth": "depth_slider", "n_out": "n_out_slider", "epochs": "epochs_slider", "nu": "nu_slider", "activation": "activation_select", "bias": "bias_toggle", "scaling": "scaling_select", "net_seed": "net_seed_input", "threshold_kind": "threshold_kind_select",
               "quantile": "quantile_slider", "share": "share_slider", "seed": "seed_input"}
# Regler, die je nach Einstellung ausgeblendet sind (Abstand bei "Lücke" und "Korrelationsbruch"; chi²-Quantil beim erwarteten Anteil; angenommener Anteil bei der Standardschwelle): der Wert bleibt hier erhalten
KEPT = {"strength_slider": "_strength_kept", "quantile_slider": "_quantile_kept", "share_slider": "_share_kept"}
PRESET_KEPT = {"strength": "_strength_kept", "quantile": "_quantile_kept", "share": "_share_kept"}


def kind_options(n_modes):
    """Die Art 'in der Lücke' gibt es erst ab zwei Betriebsarten."""
    return tuple(k for k in C.KINDS if k != "gap" or n_modes >= 2)


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; ausgeblendete Regler kehren zum zuletzt gewählten Wert zurück; eine ungültige Art wird zurückgesetzt."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default
    if st.session_state["kind_select"] not in kind_options(st.session_state["n_modes_slider"]):
        st.session_state["kind_select"] = C.DEFAULT_KIND


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in (("curvature_slider", 4), ("noise_slider", 20), ("strength_slider", 2)):
        st.session_state[key] = round(st.session_state.get(key, SETTING_SPECS[key].default) * step) / step
    st.session_state["quantile_slider"] = round(st.session_state.get("quantile_slider", C.DEFAULT_QUANTILE) * 1000) / 1000
    st.session_state["nu_slider"] = round(st.session_state.get("nu_slider", C.DEFAULT_NU) * 100) / 100
    st.session_state["width_select"] = _width(st.session_state.get("width_select", C.DEFAULT_WIDTH))
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    for key, kept in PRESET_KEPT.items():
        st.session_state[kept] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)


def randomize_net_seed():
    st.session_state["net_seed_input"] = random.randint(0, 1_000_000)
