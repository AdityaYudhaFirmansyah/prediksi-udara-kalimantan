"""Fungsi pendukung aplikasi: transformasi fitur dan pembuatan fitur dari CSV unggahan."""
import numpy as np
import pandas as pd

NAMA_PROV = {
    "West Kalimantan": "Kalimantan Barat",
    "Central Kalimantan": "Kalimantan Tengah",
    "South Kalimantan": "Kalimantan Selatan",
    "East Kalimantan": "Kalimantan Timur",
    "North Kalimantan": "Kalimantan Utara",
}
SINGKAT = {"West Kalimantan": "kalbar", "Central Kalimantan": "kalteng", "South Kalimantan": "kalsel",
           "East Kalimantan": "kaltim", "North Kalimantan": "kaltara"}
ALIAS = {}
for en, idn in NAMA_PROV.items():
    for a in (en, idn, SINGKAT[en]):
        ALIAS[a.lower().replace(" ", "")] = en

KOMPONEN = ["organic_matter", "black_carbon", "dust", "sea_salt", "sulfate"]
WAJIB = ["date", "province", "pm25", "n_hotspot"]
OPSIONAL = ["n_high", "frp_sum", "frp_max"] + KOMPONEN


def siapkan(df_mentah, cfg):
    """Fitur mentah -> masukan model: log(1+x) pada fitur hotspot dan one-hot provinsi."""
    x = df_mentah.copy()
    for c in cfg["kolom_log"]:
        x[c] = np.log1p(x[c])
    for p in NAMA_PROV:
        x[f"prov_{p}"] = (x["province"] == p).astype(int)
    return x[cfg["fitur"]]


def pangsa_komponen(df):
    return {c: float((df[c] / df["pm25"]).median()) for c in KOMPONEN}


def rasio_frp(df):
    ada = df[df.n_hotspot > 0]
    return {"frp_sum": float((ada.frp_sum / ada.n_hotspot).median()),
            "frp_max": float(ada.frp_max.median())}


def bangun_fitur(raw, pangsa, rasio):
    """Ubah CSV harian (date, province, pm25, n_hotspot, ...) menjadi fitur model.

    Mengembalikan (tabel_fitur, info). Baris yang jendela 7 harinya belum lengkap berisi NaN.
    """
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    hilang = [c for c in WAJIB if c not in raw.columns]
    if hilang:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(hilang))
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["date"].isna().any():
        raise ValueError("Ada tanggal yang tidak terbaca. Gunakan format YYYY-MM-DD.")
    kunci = raw["province"].astype(str).str.lower().str.replace(" ", "", regex=False)
    raw["province"] = kunci.map(ALIAS)
    if raw["province"].isna().any():
        tak = sorted(set(kunci[raw["province"].isna()]))
        raise ValueError("Provinsi tidak dikenali: " + ", ".join(tak) + ". Gunakan lima provinsi Kalimantan.")
    for c in ["pm25", "n_hotspot"] + [o for o in OPSIONAL if o in raw.columns]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if (raw[["pm25", "n_hotspot"]] < 0).any().any():
        raise ValueError("Nilai pm25 dan n_hotspot tidak boleh negatif.")
    if raw.duplicated(["province", "date"]).any():
        raise ValueError("Ada baris ganda untuk pasangan provinsi dan tanggal yang sama.")

    potongan, celah = [], 0
    for prov, g in raw.groupby("province"):
        g = g.sort_values("date").set_index("date")
        penuh = pd.date_range(g.index.min(), g.index.max(), freq="D")
        celah += len(penuh) - len(g)
        g = g.reindex(penuh)
        g["province"] = prov
        potongan.append(g.rename_axis("date").reset_index())
    d = pd.concat(potongan, ignore_index=True)

    diestimasi = []
    if "n_high" not in d: d["n_high"] = 0.0; diestimasi.append("n_high (diisi 0)")
    if "frp_sum" not in d: d["frp_sum"] = d["n_hotspot"] * rasio["frp_sum"]; diestimasi.append("frp_sum")
    if "frp_max" not in d: d["frp_max"] = np.where(d["n_hotspot"] > 0, rasio["frp_max"], 0.0); diestimasi.append("frp_max")
    for c in KOMPONEN:
        if c not in d: d[c] = d["pm25"] * pangsa[c]; diestimasi.append(c)

    d = d.sort_values(["province", "date"]).reset_index(drop=True)
    g = d.groupby("province", group_keys=False)
    for k in (1, 2, 3):
        d[f"hotspot_lag{k}"] = g["n_hotspot"].shift(k)
        d[f"pm25_lag{k}"] = g["pm25"].shift(k)
    d["hotspot_roll7"] = g["n_hotspot"].transform(lambda s: s.rolling(7).sum())
    d["pm25_roll7"] = g["pm25"].transform(lambda s: s.rolling(7).mean())
    d["om_share"] = d["organic_matter"] / d["pm25"].replace(0, np.nan)
    d["bulan"] = d["date"].dt.month
    return d, {"diestimasi": diestimasi, "celah": int(celah)}


def prediksi_batch(fitur, model, cfg):
    """Prediksi untuk setiap baris yang fiturnya lengkap. Mengembalikan tabel dengan peluang dan kelas."""
    kolom = [c for c in cfg["fitur"] if not c.startswith("prov_")]
    ok = fitur.dropna(subset=kolom).copy()
    if ok.empty:
        return ok
    x = siapkan(ok, cfg)
    ok["peluang"] = model.predict_proba(x)[:, 1]
    ok["kelas"] = model.predict(x).astype(int)
    ok["tanggal_prediksi"] = ok["date"] + pd.Timedelta(days=1)
    return ok
