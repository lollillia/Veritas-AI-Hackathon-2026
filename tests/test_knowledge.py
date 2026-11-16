import pytest

from app.knowledge.loader import DiseaseInfo, KnowledgeError, WeatherRule, load_knowledge

SRC = "https://example.org/rust"
ABOUT = " ".join(["word"] * 45)
LABELS = ("healthy", "common_rust")


def _files(tmp_path, yaml_text, sources=(SRC,)):
    y = tmp_path / "diseases.yaml"
    y.write_text(yaml_text)
    md = tmp_path / "SOURCES.md"
    md.write_text("\n".join(f"- {s} (used)" for s in sources))
    return y, md


GOOD = f"""
healthy:
  name: Healthy leaf
  pathogen: none
  about: {ABOUT}
  sources: [{SRC}]
common_rust:
  name: Common rust
  pathogen: Puccinia sorghi
  about: {ABOUT}
  sources: [{SRC}]
  regions:
    US: {{level: common, source: {SRC}}}
    US-FL: {{level: rare, source: {SRC}}}
    KE: {{level: unknown}}
  season: {{months: [6, 7, 8], source: {SRC}}}
  weather: {{temp_c: [16, 23], rh_min_pct: 85, source: {SRC}}}
"""


def test_valid_file_loads(tmp_path):
    k = load_knowledge(*_files(tmp_path, GOOD), labels=LABELS)
    rust = k["common_rust"]
    assert isinstance(rust, DiseaseInfo) and rust.name == "Common rust" and rust.pathogen == "Puccinia sorghi"
    assert rust.regions == {"US": ("common", SRC), "US-FL": ("rare", SRC), "KE": ("unknown", "")}
    assert rust.season_months == [6, 7, 8] and rust.season_source == SRC
    assert rust.weather == WeatherRule(16.0, 23.0, 85.0, None, SRC)
    assert k["healthy"].regions == {} and k["healthy"].weather is None and k["healthy"].season_months is None


def test_missing_label_is_reported(tmp_path):
    with pytest.raises(KnowledgeError, match="missing: common_rust"):
        load_knowledge(*_files(tmp_path, GOOD.split("common_rust:")[0]), labels=LABELS)


def test_bad_level_month_and_temp_range_are_all_reported(tmp_path):
    bad = GOOD.replace("level: common", "level: everywhere").replace("[6, 7, 8]", "[6, 13]").replace("[16, 23]", "[23, 16]")
    with pytest.raises(KnowledgeError) as exc:
        load_knowledge(*_files(tmp_path, bad), labels=LABELS)
    msg = str(exc.value)
    assert "everywhere" in msg and "13" in msg and "temp_c" in msg


def test_unsourced_entry_is_reported(tmp_path):
    bad = GOOD.replace(f"US: {{level: common, source: {SRC}}}", "US: {level: common}")
    with pytest.raises(KnowledgeError, match="US.*source"):
        load_knowledge(*_files(tmp_path, bad), labels=LABELS)


def test_source_not_in_sources_md_is_reported(tmp_path):
    with pytest.raises(KnowledgeError, match="not in SOURCES.md"):
        load_knowledge(*_files(tmp_path, GOOD, sources=("https://other.org",)), labels=LABELS)


def test_short_about_is_reported(tmp_path):
    bad = GOOD.replace(f"  pathogen: Puccinia sorghi\n  about: {ABOUT}", "  pathogen: Puccinia sorghi\n  about: too short")
    with pytest.raises(KnowledgeError, match="about"):
        load_knowledge(*_files(tmp_path, bad), labels=LABELS)


def test_weather_needs_a_moisture_condition(tmp_path):
    bad = GOOD.replace("rh_min_pct: 85, ", "")
    with pytest.raises(KnowledgeError, match="moisture"):
        load_knowledge(*_files(tmp_path, bad), labels=LABELS)


def test_healthy_must_not_have_factors(tmp_path):
    bad = GOOD.replace(f"healthy:\n  name: Healthy leaf\n", f"healthy:\n  name: Healthy leaf\n  season: {{months: [7], source: {SRC}}}\n")
    with pytest.raises(KnowledgeError, match="healthy"):
        load_knowledge(*_files(tmp_path, bad), labels=LABELS)
