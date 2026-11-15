import csv
from pathlib import Path

from PIL import Image

from data_prep.image_metadata import COLUMNS, read_photo_metadata, write_metadata_csv


def _jpeg_with_exif(path: Path, when=None, gps=None, make=None, model=None):
    exif = Image.Exif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if when:
        exif.get_ifd(0x8769)[0x9003] = when
    if gps:
        exif.get_ifd(0x8825).update(gps)
    Image.new("RGB", (16, 16), (0, 120, 0)).save(path, format="JPEG", exif=exif)


def test_reads_capture_time_gps_and_camera(tmp_path):
    p = tmp_path / "a.jpg"
    _jpeg_with_exif(p, when="2020:08:16 10:22:31", make="Apple", model="iPhone 11 Pro",
                    gps={1: "N", 2: (40.0, 30.0, 0.0), 3: "W", 4: (87.0, 1.0, 30.0)})
    m = read_photo_metadata(p)
    assert m["capture_datetime"] == "2020-08-16T10:22:31"
    assert m["gps_lat"] == 40.5 and m["gps_lon"] == -87.025      # rounded to 3 decimals
    assert m["camera"] == "Apple iPhone 11 Pro"


def test_southern_and_eastern_hemispheres_have_the_right_signs(tmp_path):
    p = tmp_path / "k.jpg"
    _jpeg_with_exif(p, gps={1: "S", 2: (0.0, 24.0, 0.0), 3: "E", 4: (35.0, 12.0, 0.0)})
    m = read_photo_metadata(p)
    assert m["gps_lat"] == -0.4 and m["gps_lon"] == 35.2


def test_image_without_exif_gives_blanks(tmp_path):
    p = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(p)
    assert read_photo_metadata(p) == {"capture_datetime": "", "gps_lat": "", "gps_lon": "", "camera": ""}


def test_falls_back_to_the_file_datetime_tag(tmp_path):
    p = tmp_path / "b.jpg"
    exif = Image.Exif()
    exif[0x0132] = "2007:08:10 09:00:00"
    Image.new("RGB", (8, 8)).save(p, format="JPEG", exif=exif)
    assert read_photo_metadata(p)["capture_datetime"] == "2007-08-10T09:00:00"


def test_malformed_values_give_blanks_not_errors(tmp_path):
    p = tmp_path / "bad.jpg"
    _jpeg_with_exif(p, when="not a date", gps={1: "N", 2: (0.0, 0.0, 0.0)})     # latitude without longitude
    m = read_photo_metadata(p)
    assert m["capture_datetime"] == "" and m["gps_lat"] == "" and m["gps_lon"] == ""


def test_unreadable_file_gives_blanks(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_text("not an image")
    assert read_photo_metadata(p)["capture_datetime"] == ""


def test_write_metadata_csv_one_row_per_manifest_row(tmp_path):
    img = tmp_path / "img.jpg"
    _jpeg_with_exif(img, when="2020:07:19 08:00:00")
    rows = [{"path": "img.jpg", "md5": "abc", "source": "cds"}]
    counts = write_metadata_csv(rows, tmp_path, tmp_path / "meta.csv")
    with open(tmp_path / "meta.csv", newline="") as f:
        out = list(csv.DictReader(f))
    assert list(out[0]) == COLUMNS
    assert out[0]["path"] == "img.jpg" and out[0]["capture_date"] == "2020-07-19" and out[0]["gps_lat"] == ""
    assert "capture_datetime" not in out[0]
    assert counts == {"images": 1, "capture_date": 1, "gps": 0, "camera": 0, "city": 0}


def test_placeholder_gps_with_null_refs_and_zero_denominators_is_blank(tmp_path):
    from PIL.TiffImagePlugin import IFDRational

    p = tmp_path / "placeholder.jpg"
    nan = IFDRational(0, 0)
    _jpeg_with_exif(p, gps={1: "\x00", 2: (nan, nan, nan), 3: "\x00", 4: (nan, nan, nan)})
    m = read_photo_metadata(p)
    assert m["gps_lat"] == "" and m["gps_lon"] == ""


def test_site_column_comes_from_documented_dataset_locations(tmp_path):
    from data_prep.image_metadata import SOURCE_SITES

    img = tmp_path / "img.jpg"
    _jpeg_with_exif(img)
    rows = [{"path": "img.jpg", "md5": "a", "source": "ndisan"}, {"path": "img.jpg", "md5": "b", "source": "rlnst"}]
    write_metadata_csv(rows, tmp_path, tmp_path / "meta.csv")
    with open(tmp_path / "meta.csv", newline="") as f:
        out = list(csv.DictReader(f))
    assert "site" in COLUMNS
    assert out[0]["site"] == SOURCE_SITES["ndisan"] and "Sampang" in out[0]["site"]
    assert out[1]["site"] == ""


def test_city_comes_from_photo_gps_otherwise_from_the_site(tmp_path):
    from data_prep.image_metadata import SOURCE_SITES

    gps_img, plain = tmp_path / "gps.jpg", tmp_path / "plain.jpg"
    _jpeg_with_exif(gps_img, when="2020:08:16 10:00:00", gps={1: "N", 2: (40.0, 28.0, 22.8), 3: "W", 4: (86.0, 59.0, 34.8)})
    _jpeg_with_exif(plain, when="2023:07:28 17:14:48")
    def resolve(lat, lon):
        return {"city": f"City at {lat},{lon}", "city_lat": round(lat, 1), "city_lon": round(lon, 1)}
    rows = [{"path": "gps.jpg", "md5": "a", "source": "cds"}, {"path": "plain.jpg", "md5": "b", "source": "ndisan"},
            {"path": "plain.jpg", "md5": "c", "source": "rlnst"}]
    write_metadata_csv(rows, tmp_path, tmp_path / "meta.csv", resolve_city=resolve)
    with open(tmp_path / "meta.csv", newline="") as f:
        out = list(csv.DictReader(f))
    assert out[0]["city"] == "City at 40.473,-86.993"
    assert out[1]["city"] == "City at -7.205,113.247" and out[1]["site"] == SOURCE_SITES["ndisan"]
    assert out[2]["city"] == "" and out[2]["site"] == ""
    assert out[0]["capture_date"] == "2020-08-16" and out[1]["capture_date"] == "2023-07-28"
    assert COLUMNS[-2:] == ["site", "city"]
    assert not {"capture_datetime", "city_lat", "city_lon", "city_basis", "site_lat", "site_lon", "site_basis"} & set(COLUMNS)


def test_per_folder_copies_split_the_metadata_by_raw_dataset_folder(tmp_path):
    from data_prep.image_metadata import write_per_folder

    rows = [{c: "" for c in COLUMNS} | {"path": p, "md5": m, "source": s} for p, m, s in [
        ("data/raw/cds/train/gls/a.jpg", "1", "cds"), ("data/raw/cds/test/nlb/b.jpg", "2", "cds"),
        ("data/raw/Final corn dataset/Healthy/c.jpg", "3", "mendeley_bd"),
        ("data/raw/gone/x.jpg", "4", "old")]]                     # folder no longer on disk: skipped
    for folder in ("cds", "Final corn dataset"):
        (tmp_path / "data" / "raw" / folder).mkdir(parents=True)
    written = write_per_folder(rows, tmp_path / "data" / "raw")
    assert written == {"cds": 2, "Final corn dataset": 1}
    with open(tmp_path / "data" / "raw" / "cds" / "image_metadata.csv", newline="") as f:
        out = list(csv.DictReader(f))
    assert [r["md5"] for r in out] == ["1", "2"] and list(out[0]) == COLUMNS
    assert (tmp_path / "data" / "raw" / "Final corn dataset" / "image_metadata.csv").is_file()
    assert not (tmp_path / "data" / "raw" / "gone").exists()


def test_label_column_comes_from_the_manifest(tmp_path):
    img = tmp_path / "img.jpg"
    _jpeg_with_exif(img)
    rows = [{"path": "img.jpg", "md5": "a", "source": "up_field", "label": "phaeosphaeria_leaf_spot"}]
    write_metadata_csv(rows, tmp_path, tmp_path / "meta.csv")
    with open(tmp_path / "meta.csv", newline="") as f:
        out = list(csv.DictReader(f))
    assert out[0]["label"] == "phaeosphaeria_leaf_spot"
    assert COLUMNS[:4] == ["path", "md5", "source", "label"]
