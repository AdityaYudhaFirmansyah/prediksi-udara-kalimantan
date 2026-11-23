import io
import json
from pathlib import Path
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import (NAMA_PROV, KOMPONEN, OPSIONAL, WAJIB, siapkan, pangsa_komponen, rasio_frp,
                   bangun_fitur, prediksi_batch)

BASE = Path(__file__).resolve().parent   # semua berkas dicari relatif terhadap lokasi app.py

st.set_page_config(page_title="Prediksi Udara Kalimantan", page_icon="🌫️", layout="wide")

# ------------------------------------------------------------------ gaya (kartu gelap, responsif)
CSS = """
<style>
#MainMenu, footer {visibility:hidden}
header[data-testid="stHeader"] {background:transparent}
[data-testid="stToolbar"], .stAppDeployButton, [data-testid="stDecoration"] {display:none !important}
.block-container {padding-top:1.4rem; padding-bottom:3rem; max-width:1080px}
@media (max-width:640px){ .block-container {padding-left:.9rem; padding-right:.9rem} [role="tablist"] [role="tab"] {padding:0 9px !important} [role="tablist"] [role="tab"] p {font-size:14px !important} }


[role="tablist"] {gap:4px; background:#171A22 !important; padding:4px !important; border-radius:14px !important; overflow-x:auto; border-bottom:none !important; box-shadow:none !important}
[role="tablist"] [role="tab"] {height:42px; border-radius:10px !important; padding:0 14px; color:#9AA4B2; white-space:nowrap}
[role="tablist"] [role="tab"][aria-selected="true"] {background:#2C313C !important; color:#fff !important}
[role="tablist"] [role="tab"] p {font-size:15px}
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display:none !important}

.app-title {font-size:28px; font-weight:700; margin:4px 0 2px; letter-spacing:-.5px}
.app-sub {color:#9AA4B2; font-size:14px; margin-bottom:14px}
.sec {font-size:20px; font-weight:700; margin:22px 0 10px}

.card {background:#171A22; border-radius:20px; padding:18px 20px; margin-bottom:12px}
.grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:12px}
@media (min-width:900px){ .grid.g4 {grid-template-columns:repeat(4,minmax(0,1fr))} .grid.g5 {grid-template-columns:repeat(5,minmax(0,1fr))} }
.tile {background:#171A22; border-radius:20px; padding:16px 18px}
.ic {width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:20px; margin-bottom:12px}
.val {font-size:30px; font-weight:700; line-height:1.1; letter-spacing:-.5px}
.u {font-size:15px; font-weight:600; color:#9AA4B2; margin-left:4px}
.lab {color:#9AA4B2; font-size:14px; margin-top:4px}
.sm {font-size:13px; color:#9AA4B2}

.hero {display:flex; gap:22px; align-items:center; flex-wrap:wrap}
.gauge {--p:0; --c:#6FCF97; width:132px; height:132px; border-radius:50%; flex:none;
        background:conic-gradient(var(--c) calc(var(--p)*1%), #2C313C 0); display:flex; align-items:center; justify-content:center}
.gauge-in {width:104px; height:104px; border-radius:50%; background:#171A22; display:flex; flex-direction:column; align-items:center; justify-content:center}
.gauge-in b {font-size:30px; line-height:1}
.gauge-in span {font-size:12px; color:#9AA4B2; margin-top:4px}
.chip {display:inline-flex; align-items:center; gap:6px; font-weight:700; font-size:15px; padding:4px 12px; border-radius:999px}
.chip.good {background:rgba(111,207,151,.15); color:#6FCF97}
.chip.bad {background:rgba(255,107,107,.16); color:#FF6B6B}
.hero-t {font-size:20px; font-weight:700; margin:10px 0 4px; line-height:1.25}
.pill {display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:10px}
.note {background:rgba(242,166,90,.10); border:1px solid rgba(242,166,90,.35); color:#F3D3AE; border-radius:16px; padding:14px 16px; font-size:14px; line-height:1.5; margin:10px 0}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:20px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ data dan model
@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model_rf_karhutla.joblib"), json.load(open(BASE / "fitur_model.json"))


@st.cache_data
def muat_data():
    df = pd.read_csv(BASE / "dataset_karhutla.csv", parse_dates=["date"])
    return df, pangsa_komponen(df), rasio_frp(df)


model, cfg = muat_model()
data, PANGSA, RASIO = muat_data()
BATAS = cfg["batas_baik"]


def idn(x, d=0):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tile(icon, warna, nilai, satuan, label):
    return (f'<div class="tile"><div class="ic" style="background:{warna}">{icon}</div>'
            f'<div class="val">{nilai}<span class="u">{satuan}</span></div><div class="lab">{label}</div></div>')


def hero(prob, kelas, judul=None, sub=""):
    bad = kelas == 1
    warna = "#FF6B6B" if bad else "#6FCF97"
    chip = ('<span class="chip bad">✕ Tidak Baik</span>' if bad else '<span class="chip good">✓ Baik</span>')
    judul = judul or (f"PM2.5 besok diperkirakan di atas {idn(BATAS,1)} µg/m³" if bad
                      else f"PM2.5 besok diperkirakan di bawah {idn(BATAS,1)} µg/m³")
    st.markdown(
        f'<div class="card hero"><div class="gauge" style="--p:{prob*100:.0f};--c:{warna}"><div class="gauge-in">'
        f'<b>{prob*100:.0f}%</b><span>peluang</span></div></div><div style="flex:1;min-width:200px">{chip}'
        f'<div class="hero-t">{judul}</div><div class="sm">{sub or "Peluang kelas Tidak Baik (di atas batas kategori Baik ISPU)."}</div></div></div>',
        unsafe_allow_html=True)


def tema(ch, tinggi=220):
    return (ch.properties(height=tinggi, width="container")
            .configure_axis(labelColor="#9AA4B2", titleColor="#9AA4B2", gridColor="#262A33", domainColor="#262A33", tickColor="#262A33")
            .configure_view(strokeWidth=0).configure(background="transparent"))


def grafik_pm25(j):
    d = j.reset_index()
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b"))
    garis = alt.Chart(d).mark_line(color="#F2A65A", strokeWidth=3).encode(x=x, y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"))
    area = alt.Chart(d).mark_area(opacity=.15, color="#F2A65A").encode(x=x, y="pm25:Q")
    batas = alt.Chart(pd.DataFrame({"y": [BATAS]})).mark_rule(color="#FF6B6B", strokeDash=[6, 4]).encode(y="y:Q")
    return tema(area + garis + batas)


def grafik_batang(j, kolom, warna, judul):
    d = j.reset_index()
    ch = alt.Chart(d).mark_bar(color=warna, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")), y=alt.Y(f"{kolom}:Q", title=judul))
    return tema(ch)


# ------------------------------------------------------------------ kepala halaman
st.markdown('<div class="app-title">🌫️ Prediksi Udara Besok</div>'
            f'<div class="app-sub">Peluang PM2.5 besok melewati {idn(BATAS,1)} µg/m³ di lima provinsi Kalimantan, dari hotspot karhutla dan kondisi udara terkini.</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📅 Demo", "✍️ Manual", "📂 CSV", "ℹ️ Model"])

# ================================================================== TAB 1: DEMO
with tab1:
    c1, c2 = st.columns(2)
    prov = c1.selectbox("Provinsi", list(NAMA_PROV), format_func=NAMA_PROV.get, key="demo_prov")
    sub = data[data.province == prov].sort_values("date")
    tgl = c2.date_input("Tanggal (hari ini)", value=sub.date.max().date(), min_value=sub.date.min().date(),
                        max_value=sub.date.max().date(), key="demo_tgl")
    baris = sub[sub.date == pd.Timestamp(tgl)]
    if baris.empty:
        st.warning("Tanggal tidak ada pada data.")
    else:
        b = baris.iloc[0]
        x = siapkan(baris, cfg)
        prob, kelas = float(model.predict_proba(x)[:, 1][0]), int(model.predict(x)[0])
        hero(prob, kelas)
        aktual = "Tidak Baik" if b.kelas_next == 1 else "Baik"
        st.markdown(f'<div class="sm" style="margin:-4px 0 14px 4px">Kenyataan besok (data reanalisis): PM2.5 {idn(b.pm25_next,1)} µg/m³ → <b>{aktual}</b></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="grid g4">'
                    + tile("💨", "#0F6B73", idn(b.pm25, 1), "µg/m³", "PM2.5 hari ini")
                    + tile("🔥", "#B3402F", idn(b.n_hotspot), "titik", "Hotspot hari ini")
                    + tile("📅", "#3B4BC8", idn(b.hotspot_roll7), "titik", "Hotspot 7 hari")
                    + tile("🌫️", "#7A3FB5", idn(b.pm25_lag1, 1), "µg/m³", "PM2.5 kemarin") + '</div>', unsafe_allow_html=True)
        j = sub[(sub.date <= pd.Timestamp(tgl)) & (sub.date > pd.Timestamp(tgl) - pd.Timedelta(days=45))].set_index("date")
        st.markdown('<div class="sec">45 hari terakhir</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="sm">PM2.5 (garis merah putus-putus: batas Baik)</div>', unsafe_allow_html=True)
            st.altair_chart(grafik_pm25(j))
        with st.container(border=True):
            st.markdown('<div class="sm">Jumlah hotspot harian</div>', unsafe_allow_html=True)
            st.altair_chart(grafik_batang(j, "n_hotspot", "#4FB3BF", "Hotspot"))
    st.caption("Data tersedia sampai 30 Agustus 2026 dan berasal dari data pelatihan dan pengujian model.")

# ================================================================== TAB 2: MANUAL
with tab2:
    a, b_ = st.columns(2)
    prov_m = a.selectbox("Provinsi", list(NAMA_PROV), format_func=NAMA_PROV.get, key="man_prov")
    bulan = b_.selectbox("Bulan", list(range(1, 13)), index=7, key="man_bulan")
    st.markdown('<div class="sec">PM2.5 (µg/m³)</div>', unsafe_allow_html=True)
    p0, p1, p2, p3, p7 = st.columns(5)
    pm0 = p0.number_input("Hari ini", 0.0, 500.0, 8.0, 0.5)
    pm1 = p1.number_input("Kemarin", 0.0, 500.0, 8.0, 0.5)
    pm2 = p2.number_input("2 hari lalu", 0.0, 500.0, 8.0, 0.5)
    pm3 = p3.number_input("3 hari lalu", 0.0, 500.0, 8.0, 0.5)
    pm7 = p7.number_input("Rata-rata 7 hari", 0.0, 500.0, 8.0, 0.5)
    st.markdown('<div class="sec">Hotspot (titik per hari di provinsi)</div>', unsafe_allow_html=True)
    h0, h1, h2, h3, h7 = st.columns(5)
    ht0 = h0.number_input("Hari ini ", 0, 20000, 5, 1)
    ht1 = h1.number_input("Kemarin ", 0, 20000, 5, 1)
    ht2 = h2.number_input("2 hari lalu ", 0, 20000, 5, 1)
    ht3 = h3.number_input("3 hari lalu ", 0, 20000, 5, 1)
    ht7 = h7.number_input("Total 7 hari", 0, 100000, 30, 1)
    with st.expander("Pengaturan lanjutan (intensitas hotspot dan komposisi aerosol)"):
        f1, f2, f3 = st.columns(3)
        frp_sum = f1.number_input("Total FRP hari ini (MW)", 0.0, 100000.0, float(ht0) * RASIO["frp_sum"], 1.0)
        frp_max = f2.number_input("FRP maksimum hari ini (MW)", 0.0, 2000.0, RASIO["frp_max"] if ht0 else 0.0, 1.0)
        n_high = f3.number_input("Hotspot confidence tinggi", 0, 20000, 0, 1)
        manual = st.checkbox("Isi komposisi aerosol sendiri")
        if manual:
            k = st.columns(5)
            komp = {c: k[i].number_input(c.replace("_", " "), 0.0, 500.0, round(pm0 * PANGSA[c], 2), 0.1) for i, c in enumerate(KOMPONEN)}
        else:
            komp = {c: pm0 * PANGSA[c] for c in KOMPONEN}
            st.caption("Jika tidak diisi, komposisi aerosol diperkirakan dari PM2.5 hari ini (proporsi median data pelatihan).")
    if st.button("Prediksi", type="primary", use_container_width=True):
        baris_m = pd.DataFrame([{
            "province": prov_m, "pm25": pm0, "pm25_lag1": pm1, "pm25_lag2": pm2, "pm25_lag3": pm3, "pm25_roll7": pm7,
            "n_hotspot": ht0, "hotspot_lag1": ht1, "hotspot_lag2": ht2, "hotspot_lag3": ht3, "hotspot_roll7": ht7,
            "n_high": n_high, "frp_sum": frp_sum, "frp_max": frp_max, **komp,
            "om_share": komp["organic_matter"] / pm0 if pm0 > 0 else 0.0, "bulan": bulan}])
        x = siapkan(baris_m, cfg)
        hero(float(model.predict_proba(x)[:, 1][0]), int(model.predict(x)[0]))
        if ht7 < ht0:
            st.warning("Total hotspot 7 hari lebih kecil daripada hotspot hari ini. Periksa kembali isian Anda.")

# ================================================================== TAB 3: UNGGAH CSV
with tab3:
    st.markdown(
        '<div class="card"><b>Prediksi dari data Anda sendiri</b><div class="sm" style="margin-top:6px;line-height:1.5">'
        'Unggah CSV harian per provinsi. Aplikasi menghitung lag dan rolling 7 hari otomatis, lalu memprediksi hari berikutnya '
        'untuk setiap tanggal, termasuk <b>besok dari tanggal terakhir</b> di berkas Anda. Perlu minimal 7 hari berurutan per provinsi.<br><br>'
        '<b>Kolom wajib:</b> <code>date</code> (YYYY-MM-DD), <code>province</code>, <code>pm25</code> (µg/m³), <code>n_hotspot</code> (jumlah hotspot valid per hari).<br>'
        '<b>Kolom opsional:</b> <code>n_high</code>, <code>frp_sum</code>, <code>frp_max</code>, <code>organic_matter</code>, <code>black_carbon</code>, '
        '<code>dust</code>, <code>sea_salt</code>, <code>sulfate</code>. Kolom yang tidak ada diperkirakan dari data pelatihan, sehingga hasilnya kurang akurat.'
        '</div></div>', unsafe_allow_html=True)
    contoh = (data[data.province == "West Kalimantan"].sort_values("date").tail(14)[WAJIB].copy())
    contoh["date"] = contoh["date"].dt.strftime("%Y-%m-%d")
    st.download_button("⬇️ Unduh contoh CSV", contoh.to_csv(index=False).encode("utf-8"), "contoh_input.csv", "text/csv")
    st.markdown('<div class="note">⚠️ Skala PM2.5 penting. Model dilatih dengan PM2.5 reanalisis MERRA-2 rata-rata provinsi. '
                'Nilai dari stasiun ukur biasanya lebih tinggi dan lebih ekstrem, sehingga prediksi bisa melenceng. '
                'Hotspot sebaiknya dari sensor dan penyaringan yang sama (VIIRS S-NPP, tipe 0, confidence bukan rendah).</div>', unsafe_allow_html=True)
    berkas = st.file_uploader("Pilih berkas CSV", type=["csv"])
    if berkas is not None:
        try:
            raw = pd.read_csv(berkas)
            fitur, info = bangun_fitur(raw, PANGSA, RASIO)
            hasil = prediksi_batch(fitur, model, cfg)
            if hasil.empty:
                st.error("Belum ada baris yang bisa diprediksi. Setiap provinsi memerlukan minimal 7 hari berurutan.")
            else:
                if info["diestimasi"]:
                    st.info("Kolom berikut tidak ada di berkas dan diperkirakan: " + ", ".join(info["diestimasi"]) + ".")
                if info["celah"]:
                    st.warning(f"Ditemukan {info['celah']} hari yang hilang di antara tanggal berkas. Baris yang jendela 7 harinya tidak lengkap dilewati.")
                terakhir = hasil.sort_values("date").groupby("province").tail(1).sort_values("province")
                st.markdown('<div class="sec">Prediksi untuk besok</div>', unsafe_allow_html=True)
                kartu = ""
                for _, r in terakhir.iterrows():
                    bad = r.kelas == 1
                    kartu += (f'<div class="tile"><div class="sm">{NAMA_PROV[r.province]}</div>'
                              f'<div class="val" style="color:{"#FF6B6B" if bad else "#6FCF97"}">{r.peluang*100:.0f}<span class="u">%</span></div>'
                              f'<div class="pill"><span class="chip {"bad" if bad else "good"}">{"Tidak Baik" if bad else "Baik"}</span></div>'
                              f'<div class="sm" style="margin-top:8px">untuk {r.tanggal_prediksi:%d %b %Y}</div></div>')
                st.markdown(f'<div class="grid g5">{kartu}</div>', unsafe_allow_html=True)

                pilih = st.selectbox("Lihat riwayat peluang provinsi", sorted(hasil.province.unique()),
                                     format_func=NAMA_PROV.get, key="csv_prov")
                h = hasil[hasil.province == pilih]
                with st.container(border=True):
                    st.markdown('<div class="sm">Peluang Tidak Baik untuk hari berikutnya</div>', unsafe_allow_html=True)
                    d = h.assign(peluang_pct=h.peluang * 100)
                    ch = (alt.Chart(d).mark_area(line={"color": "#F2A65A", "strokeWidth": 3}, color="#F2A65A", opacity=.2)
                          .encode(x=alt.X("tanggal_prediksi:T", title=None, axis=alt.Axis(format="%d %b")),
                                  y=alt.Y("peluang_pct:Q", title="Peluang (%)", scale=alt.Scale(domain=[0, 100]))))
                    st.altair_chart(tema(ch))
                tampil = hasil[["tanggal_prediksi", "province", "pm25", "n_hotspot", "peluang", "kelas"]].copy()
                tampil["province"] = tampil["province"].map(NAMA_PROV)
                tampil["kelas"] = np.where(tampil["kelas"] == 1, "Tidak Baik", "Baik")
                tampil["peluang"] = (tampil["peluang"] * 100).round(0).astype(int).astype(str) + "%"
                tampil.columns = ["Tanggal prediksi", "Provinsi", "PM2.5 hari ini", "Hotspot hari ini", "Peluang Tidak Baik", "Prediksi"]
                with st.expander(f"Tabel hasil ({len(tampil)} baris)"):
                    st.dataframe(tampil, hide_index=True)
                st.download_button("⬇️ Unduh hasil (CSV)", tampil.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Berkas tidak dapat diproses: {e}")

# ================================================================== TAB 4: MODEL
with tab4:
    st.markdown(
        '<div class="card"><b>Tentang model</b><div class="sm" style="margin-top:6px;line-height:1.6">'
        f'Memprediksi apakah PM2.5 provinsi <b>besok</b> melewati {idn(BATAS,1)} µg/m³ (batas kategori Baik ISPU). '
        'Data: hotspot VIIRS S-NPP (NASA FIRMS) digabung PM2.5 reanalisis MERRA-2 per provinsi Kalimantan, 7 Jan 2025 – 30 Agu 2026. '
        'Model Random Forest dipilih dari tiga algoritma (Decision Tree, Random Forest, SVM).</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Perbandingan pada data uji (1 Jun – 30 Agu 2026)</div>', unsafe_allow_html=True)
    st.dataframe(pd.read_csv(BASE / "hasil_perbandingan_model.csv").round(3), hide_index=True)
    st.markdown('<div class="note"><b>Keterbatasan.</b> Model hanya sedikit lebih baik daripada aturan sederhana “besok sama dengan hari ini” pada F1; '
                'nilai tambahnya ada pada peringatan awal lonjakan. Data latih hanya memuat 41 hari Tidak Baik, PM2.5 berasal dari reanalisis '
                'rata-rata provinsi (bukan stasiun), dan hotspot belum tentu berarti kebakaran. Aplikasi ini proyek pembelajaran dan '
                '<b>bukan pengganti informasi resmi</b> BMKG atau KLH.</div>', unsafe_allow_html=True)
