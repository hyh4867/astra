"""Local Streamlit interface for public-image change detection."""
from __future__ import annotations

import csv
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium

from inference import (
    ROOT, changed_object_summary, filter_small_components, load_tinycd_model, make_overlay,
    opencv_predictions, probability_heatmap, quality_checks, tinycd_probability,
)
from geospatial import align_pair, comparison_slider, object_features, read_image_bytes

OUT = ROOT / "outputs"
TEST_DATA = ROOT / "data" / "benchmark" / "test"
NAIP_DATA = ROOT / "data" / "demo_naip" / "celina_town_2020_2022_overlap"
SCREENSHOT_HIDE_MAP = os.environ.get("ASTRA_SCREENSHOT_HIDE_MAP") == "1"
METHODS = ["OpenCV basic", "OpenCV improved", "TinyCD"]

st.set_page_config(page_title="ASTRA | Satellite Change Detection", layout="wide")
LANG = st.sidebar.selectbox("Language / 언어", ["English", "한국어"], index=0, key="language")

def t(en: str, ko: str) -> str:
    return en if LANG == "English" else ko


@st.cache_resource
def cached_tinycd(weights_path: str):
    return load_tinycd_model(Path(weights_path))


@st.cache_data(show_spinner=False)
def cached_image_read(data: bytes, filename: str):
    return read_image_bytes(data, filename)


@st.cache_data(show_spinner=False)
def cached_aligned_pair(before, geo_a_key: str, _geo_a, after, geo_b_key: str, _geo_b):
    return align_pair(before, _geo_a, after, _geo_b)


def geo_cache_key(geo) -> str:
    if geo is None: return "none"
    return json.dumps({"transform": tuple(geo["transform"]), "crs": str(geo["crs"]),
        "gsd_m": geo.get("gsd_m"), "pixel_area_m2": geo.get("pixel_area_m2"),
        "utm_crs": geo.get("utm_crs")},sort_keys=True)


@st.cache_data(show_spinner=False)
def cached_probability(_model, weights_path: str, before: np.ndarray, after: np.ndarray, device_name: str):
    import torch
    return tinycd_probability(_model, before, after, torch.device(device_name))


def image_pair(source: str, before_file, after_file, local_name: str | None):
    if source == "합성 데모":
        rng = np.random.default_rng(7)
        base = rng.integers(40, 100, size=(96, 128, 3), dtype=np.uint8)
        before = base.copy(); after = base.copy()
        before[28:52, 38:65] = (105, 110, 120)
        after[28:52, 38:65] = (225, 220, 210)
        after[63:78, 86:110] = (200, 195, 185)
        return before, after, None, None
    if source == "로컬 LEVIR test pair":
        if not local_name:
            return None, None, None, None
        try:
            before = np.asarray(Image.open(TEST_DATA / "A" / local_name).convert("RGB"))
            after = np.asarray(Image.open(TEST_DATA / "B" / local_name).convert("RGB"))
            return before, after, None, 0.5
        except FileNotFoundError:
            st.warning("로컬 LEVIR test 자료가 없습니다. 공개 데이터는 별도로 내려받아야 합니다.")
            return None, None, None, None
    if source == "NAIP 공개 데모":
        a_path=NAIP_DATA/"A_before_2020.tif"; b_path=NAIP_DATA/"B_after_2022.tif"
        if not a_path.is_file() or not b_path.is_file():
            st.info("NAIP 데모 자료가 없습니다. scripts/acquire_naip_demo.py로 작은 공개 COG 창을 준비한 뒤 scripts/prepare_naip_overlap.py로 두 날짜의 유효 영역을 맞추세요.")
            return None, None, None, None
        a,geo_a=cached_image_read(a_path.read_bytes(),a_path.name)
        b,geo_b=cached_image_read(b_path.read_bytes(),b_path.name)
        a,b,geo,gsd=cached_aligned_pair(a,geo_cache_key(geo_a),geo_a,b,geo_cache_key(geo_b),geo_b)
        return a,b,geo,gsd
    if before_file is None or after_file is None:
        return None, None, None, None
    before,geo_a=cached_image_read(before_file.getvalue(),before_file.name)
    after,geo_b=cached_image_read(after_file.getvalue(),after_file.name)
    before,after,geo,gsd=cached_aligned_pair(before,geo_cache_key(geo_a),geo_a,after,geo_cache_key(geo_b),geo_b)
    return before,after,geo,gsd


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def probability_colorbar():
    values=np.arange(256,dtype=np.uint8).reshape(1,256)
    bar=cv2.cvtColor(cv2.applyColorMap(values,cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
    return np.repeat(bar,18,axis=0)


def render_results():
    st.subheader(t("Key findings", "핵심 결과"))
    st.markdown(t("""
- **Test F1 / object skill:** OpenCV basic **0.131 / 0.076**, OpenCV improved **0.100 / 0.009**, TinyCD **0.911 / 0.953**.
- **Small isolated objects:** TinyCD reaches 50% completeness at **209 px ≈ 52 m²** (at 0.5 m/px).
- **Misregistration:** Under B-only shifts, A-frame F1 falls (0.903→0.723 at 1→5 px) but B-frame F1 stays 0.920–0.921; ΔFP≈ΔFN shows a coordinate-frame artifact, not spurious detections.
- **Blur / BatchNorm:** On the same 32-pair subset, B σ=2 F1 is **0.025** for ROBUST-v1 and **0.419** when BatchNorm updates without shifts; frozen-BN variants reach **0.894–0.896**, even with the v1 shift augmentation (single-seed ablation).
- **Direction:** TinyCD favors appearance of new buildings and is unvalidated for disappearance/demolition.
""", """
- **테스트 F1 / 객체 skill:** OpenCV basic **0.131 / 0.076**, OpenCV improved **0.100 / 0.009**, TinyCD **0.911 / 0.953**.
- **작은 고립 객체:** TinyCD의 50% 완전성 크기는 **209 px ≈ 52 m²** (0.5 m/px 기준)입니다.
- **정합 민감도:** B만 이동시키면 A 프레임 F1은 1→5px에서 0.903→0.723으로 낮아지지만, B 프레임 F1은 0.920–0.921입니다. ΔFP≈ΔFN은 오탐 생성이 아니라 좌표계 불일치에 따른 현상임을 보여줍니다.
- **Blur / BatchNorm:** 같은 32쌍에서 B blur σ=2 F1은 ROBUST-v1 **0.025**, shift 없이 BatchNorm을 갱신한 경우 **0.419**였습니다. BatchNorm을 고정한 두 모델은 v1 shift를 포함해도 **0.894–0.896**을 기록했습니다(단일 seed 진단).
- **방향성:** TinyCD는 새 건물의 출현에 치우쳐 보이며 철거/소실은 검증되지 않았습니다.
"""))
    st.subheader(t("LEVIR-CD benchmark", "LEVIR-CD 벤치마크"))
    bench = read_csv(OUT / "benchmark_test.csv")
    if bench:
        rows = [r for r in bench if r["method"] in METHODS]
        table=[]
        for r in rows:
            table.append({"method":r["method"],"P":f"{float(r['precision']):.4f}",
                "R":f"{float(r['recall']):.4f}","F1":f"{float(r['f1']):.4f}",
                "IoU":f"{float(r['iou']):.4f}","runtime (s)":f"{float(r['runtime_seconds']):,.3f}",
                "device":r.get("device","")})
        mps_runtime=read_csv(OUT/"tinycd_test_mps_comparison.csv")
        if mps_runtime:
            mps_seconds=float(mps_runtime[0]["runtime_seconds"])
            for row in table:
                if row["method"]=="TinyCD":
                    row["runtime (s)"]=f"{float(row['runtime (s)']):,.3f} CPU / {mps_seconds:,.3f} MPS"
                    row["device"]="CPU / MPS"
        st.dataframe(table, hide_index=True, width="stretch")
    comp = read_csv(OUT / "object_metrics_summary.csv")
    if comp:
        st.subheader(t("Object completeness and false alarms", "객체 완전성 및 오경보"))
        table=[]
        skill={"OpenCV basic":"0.076","OpenCV improved":"0.009","TinyCD":"0.953"}
        isolated_size={"OpenCV basic":"9,658 px","OpenCV improved":"below measured range (<100 px)","TinyCD":"209 px ≈ 52 m²"}
        for r in comp:
            size="below measured range (<100 px)" if r["method"]=="TinyCD" else f"{float(r['completeness50_object_px']):,.0f} px"
            table.append({"method":r["method"],"GT objects":f"{int(r['n_gt_objects']):,}",
                "false alarms/km²":f"{float(r['false_alarms_per_km2']):,.1f}","object skill":skill[r["method"]],
                "overall 50% size":size,"isolated 50% size":isolated_size[r["method"]]})
        st.dataframe(table, hide_index=True, width="stretch")
    advanced_path=OUT/"diagnosis_round"/"round3_test_metrics.csv"
    advanced = [r for r in read_csv(advanced_path) if r["method"] in {"Original", "CONTROL-v2-42_best", "ROBUST-v2-42_best"}]
    if advanced:
        st.subheader("Fine-tuning 비교 · held-out TEST (128쌍)")
        st.caption(t("Across-seed clean VAL F1 selected CONTROL-v2 seed 42 (0.909601) and ROBUST-v2 seed 42 (0.905744). No TEST-based checkpoint selection.","seed 간 clean VAL F1로 CONTROL-v2 seed 42 (0.909601), ROBUST-v2 seed 42 (0.905744)를 선택했습니다. TEST 점수로 checkpoint를 고르지 않았습니다."))
        table=[]
        for r in advanced:
            labels={"Original":"Original","CONTROL-v2-42_best":"CONTROL-v2 seed 42","ROBUST-v2-42_best":"ROBUST-v2 seed 42"}
            table.append({"model":labels[r["method"]],"P":f"{float(r['precision']):.4f}",
                "R":f"{float(r['recall']):.4f}","F1":f"{float(r['f1']):.4f}",
                "IoU":f"{float(r['iou']):.4f}","TP":f"{int(r['tp']):,}",
                "FP":f"{int(r['fp']):,}","FN":f"{int(r['fn']):,}",
                "runtime (s)":f"{float(r['runtime_seconds']):,.1f}"})
        st.dataframe(table, hide_index=True, width="stretch")
        advanced_dir=advanced_path.parent
        swapped=read_csv(advanced_dir/"round3_swap.csv")
        if swapped:
            wanted={"Original","CONTROL-v2-42_best","ROBUST-v2-42_best"}
            swap_text=" · ".join(f"{r['method']} B→A recall/F1 {float(r['recall']):.3f}/{float(r['f1']):.3f}" for r in swapped if r["method"] in wanted)
            st.caption(t("Input order reversed (32 pairs): "+swap_text+". Direction result is a single seed-42 reference.","입력 순서 반전 (32쌍): "+swap_text+". 방향성 결과는 seed 42 단일 checkpoint 참고값입니다."))
        robust_rows=read_csv(advanced_dir/"round3_robustness.csv")
        robust_sigma2=[r for r in robust_rows if r["method"] in {"Original","CONTROL-v2-42_best","ROBUST-v2-42_best"} and r["condition"]=="blur_B" and float(r["strength"])==2 and r["frame"]=="A-frame"]
        if robust_sigma2:
            vals=" · ".join(f"{r['method']} {float(r['f1']):.3f}" for r in robust_sigma2)
            st.info(t("B blur σ=2 F1 (Original / CONTROL-v2 s42 / ROBUST-v2 s42): "+vals,"B blur σ=2 F1 (Original / CONTROL-v2 s42 / ROBUST-v2 s42): "+vals))
    naip_summary_path=OUT/"naip_demo"/"round4_naip_checkpoint_selection.json"
    if naip_summary_path.is_file():
        naip=json.loads(naip_summary_path.read_text())
        by_model={}
        for result in naip.get("models",[]):
            by_model.setdefault(result["model"],{})[result["scale"]]=result
        rows=[]
        for label in ("Original","CONTROL-v2 seed 42","ROBUST-v2 seed 42"):
            values=by_model.get(label,{})
            def count_area(scale):
                item=values.get(scale)
                return "—" if not item else f"{item['change_objects']:,} / {item['total_area_m2']:,.1f}"
            rows.append({"model":label,"0.6 m objects / m²":count_area("0p6m"),"0.5 m objects / m²":count_area("0p5m")})
        st.subheader(t("NAIP qualitative check · no ground truth","NAIP 정성 확인 · 정답 없음"))
        st.dataframe(rows,hide_index=True,width="stretch")
        st.caption(t("NAIP object counts are descriptive only and were not used to select checkpoints.","NAIP 객체 수는 설명용이며 checkpoint 선택에 사용하지 않았습니다."))
    directional_union=read_csv(OUT/"tinycd_bidirectional_union.csv")
    if directional_union:
        r=directional_union[0]
        st.info(t(f"EXPERIMENTAL bidirectional union (32 pairs, reference only): P/R/F1 {float(r['precision']):.4f}/{float(r['recall']):.4f}/{float(r['f1']):.4f}. LEVIR labels do not encode direction.",f"EXPERIMENTAL 방향 양방향 union (32쌍, 참고용): P/R/F1 {float(r['precision']):.4f}/{float(r['recall']):.4f}/{float(r['f1']):.4f}. LEVIR 정답에는 방향 정보가 없어 disappearance 검증 점수가 아닙니다."))
    mps = read_csv(OUT / "tinycd_test_mps_comparison.csv")
    if mps:
        r=mps[0]
        st.info(t(f"TinyCD CPU↔MPS TP/FP/FN match: {int(r['tp']):,}/{int(r['fp']):,}/{int(r['fn']):,}; MPS {float(r['runtime_seconds']):,.3f}s",f"TinyCD CPU↔MPS TP/FP/FN 일치: {int(r['tp']):,}/{int(r['fp']):,}/{int(r['fn']):,}; MPS {float(r['runtime_seconds']):,.3f}초"))
    blur = read_csv(OUT / "blur_breakdown_mps.csv")
    if blur:
        st.caption(t("32-pair MPS blur sweep: TinyCD F1 is 0.911 at B σ=1.50, 0.857 at σ=1.75, and 0.640 at σ=2.","32쌍 MPS blur sweep: TinyCD F1은 B blur σ=1.50에서 0.911, σ=1.75에서 0.857, σ=2에서 0.640입니다."))
    null = read_csv(OUT / "object_null_completeness_seed42.csv")
    if null:
        st.caption(t("Includes the seed-42 deranged-mask null test. TinyCD all-object completeness: real 0.957 / null 0.071.","seed-42 다른 영상 마스크 null test 포함. TinyCD 전체 객체 완전성: 실제 0.957 / null 0.071."))
    charts = [
        ("benchmark_test.png", t("Test benchmark P/R/F1/IoU","테스트 benchmark P/R/F1/IoU")),
        ("completeness_vs_size.png", t("Completeness by object size (Wilson 95% CI)","전체 객체 크기별 완전성 (Wilson 95% CI)")),
        ("isolated_completeness_vs_size.png", t("Completeness for isolated objects","격리 객체 크기별 완전성")),
        ("robustness_f1.png", t("Robustness and blur sensitivity","강건성 및 blur 민감도")),
        ("diagnosis_round/robustness_f1_advanced.png", "Unified-pipeline robustness (seed 42)"),
        ("object_null_completeness.png", t("Real pairs vs deranged-mask null","실제 pair와 deranged-mask null 비교")),
    ]
    for name, caption in charts:
        path=OUT/name
        if path.is_file(): st.image(str(path),caption=caption,width="stretch")
        else: st.caption(t(f"Result file not available yet: {name}",f"아직 결과 파일이 없습니다: {name}"))


st.title("ASTRA: AI Satellite Change Detection")
st.caption(t("A = before (older) · B = after (newer) · public satellite imagery · local processing", "A = 이전 · B = 이후 · 공개 위성영상 변화 분석 · 로컬 처리"))
tab_analyze, tab_results, tab_limits = st.tabs([t("Analyze", "변화 분석"), "Results", "Limitations"])

with tab_analyze:
    test_mode=st.query_params.get("demo")=="1" or os.environ.get("ASTRA_TEST_MODE")=="1"
    source_options=[t("NAIP public demo", "NAIP 공개 데모"),t("Upload images", "파일 업로드"),t("Local LEVIR test pair", "로컬 LEVIR test pair")]
    if test_mode:
        source_options.insert(2,t("Synthetic demo", "합성 데모"))
    demo_index=source_options.index(t("Synthetic demo", "합성 데모")) if test_mode else 0
    source_label = st.selectbox(t("Image source", "영상 입력"), source_options, index=demo_index, key="image_source")
    source = {t("NAIP public demo", "NAIP 공개 데모"):"NAIP 공개 데모",t("Upload images", "파일 업로드"):"파일 업로드",t("Local LEVIR test pair", "로컬 LEVIR test pair"):"로컬 LEVIR test pair",t("Synthetic demo", "합성 데모"):"합성 데모"}[source_label]
    if source == "NAIP 공개 데모":
        st.info(t("Qualitative demo only · USDA NAIP (Celina, TX; 2020 → 2022, 0.6 m GSD) · no ground truth", "질적 시연 전용 · USDA NAIP 공개 항공영상 (Celina, TX; 2020 → 2022, 0.6 m GSD) · 정답 없음"))
    before_file=after_file=None; local_name=None
    if source == "파일 업로드":
        c1,c2=st.columns(2)
        with c1: before_file=st.file_uploader(t("A = before (older)","A = 이전 시점 (과거)"),type=["png","jpg","jpeg","tif","tiff"],key="before_upload")
        with c2: after_file=st.file_uploader(t("B = after (newer)","B = 이후 시점 (최근)"),type=["png","jpg","jpeg","tif","tiff"],key="after_upload")
    elif source == "로컬 LEVIR test pair":
        local_names=sorted(p.name for p in (TEST_DATA/"A").glob("*.png")) if (TEST_DATA/"A").is_dir() else []
        if local_names: local_name=st.selectbox("로컬 test pair",local_names,key="local_sample")
        else: st.info("로컬 LEVIR test 이미지를 찾을 수 없습니다. 업로드 또는 합성 데모를 선택하세요.")

    method=st.selectbox(t("Detection method","탐지 방법"),METHODS,index=2,key="method")
    robust_weights=ROOT/"models"/"finetuned"/"tinycd_robust_v2_seed42_best.pth"
    model_variant="Original"
    if method=="TinyCD":
        robust_label="ROBUST-v2 (blur/order-robust fine-tune)"
        variants=["Original"]+([robust_label] if robust_weights.is_file() else [])
        model_variant=st.selectbox(t("TinyCD model","TinyCD 모델"),variants,key="model_variant")
        st.caption(t("Checkpoint selection used clean VAL F1 across seeds. Selected: CONTROL-v2 s42 0.909601; ROBUST-v2 s42 0.905744.","seed 간 clean VAL F1로 checkpoint를 선택했습니다. 선정값: CONTROL-v2 s42 0.909601; ROBUST-v2 s42 0.905744."))
    bidirectional=st.checkbox(t("EXPERIMENTAL: show appeared / possibly disappeared","실험적: 출현 / 소멸 가능성 양방향 표시"),value=False,
        key="bidirectional",disabled=method!="TinyCD" or model_variant!="Original",
        help=t("Original TinyCD only. Direction is unvalidated because LEVIR labels do not encode direction. Fine-tuned order-invariant models disable this mode.","Original TinyCD에만 적용합니다. LEVIR 정답은 방향을 담지 않아 검증되지 않았습니다. 순서 불변 fine-tuned 모델에서는 사용할 수 없습니다."))
    if method=="TinyCD" and model_variant=="Original":
        st.warning(t("Original TinyCD is blur-sensitive: F1 was 0.640 for B blur σ=2 on the 32-pair LEVIR subset. ROBUST-v2 seed 42 scored 0.896 on that condition.","Original TinyCD는 blur에 민감합니다. LEVIR 32쌍의 B blur σ=2에서 F1은 0.640이었습니다. ROBUST-v2 seed 42는 같은 조건에서 0.896을 기록했습니다."))
    try:
        before,after,geo,auto_gsd=image_pair(source,before_file,after_file,local_name)
    except Exception as exc:
        st.error(t(f"Could not read or align images: {exc}",f"영상을 읽거나 정렬하지 못했습니다: {exc}"))
        before=after=geo=auto_gsd=None
    if auto_gsd is not None:
        gsd=float(auto_gsd)
        st.metric(t("GSD read from GeoTIFF","GeoTIFF에서 읽은 GSD"),f"{gsd:.3f} m/pixel")
        if geo:
            st.caption(t(f"CRS {geo['crs']} · area CRS {geo['utm_crs']} · B aligned to A's GeoTIFF grid",f"CRS {geo['crs']} · 면적 계산 {geo['utm_crs']} · B는 A의 GeoTIFF grid로 정렬"))
    else:
        gsd=st.number_input(t("GSD (m/pixel)","GSD (m/pixel)"),min_value=0.01,max_value=100.0,value=0.5,step=0.1,key="gsd")
    minimum_object_area_m2=st.slider(
        t("Minimum object area (m²)","최소 객체 면적 (m²)"),
        min_value=0.0,max_value=500.0,value=5.0,step=0.25,key="minimum_object_area_m2",
        help=t("Default 5 m². The app derives the pixel cutoff from GSD; set 0 to keep every component.",
               "기본값 5m²입니다. GSD로 픽셀 기준을 계산하며, 0이면 모든 component를 유지합니다."))
    if before is None or after is None:
        st.info(t("Select or upload an A/B pair to see the analysis.","A/B 영상을 선택하면 분석 결과가 표시됩니다."))
    elif before.shape != after.shape:
        st.error(t(f"A and B must have matching pixel dimensions. A={before.shape[:2]}, B={after.shape[:2]}",f"A와 B는 같은 픽셀 크기여야 합니다. 현재 A={before.shape[:2]}, B={after.shape[:2]}"))
    else:
        checks=quality_checks(before,after,gsd)
        for message in checks["warnings"]:
            if LANG=="한국어":
                if message.startswith("Estimated image registration shift"):
                    message=f"추정 영상 정합 차이는 {checks['shift_px']:.1f}px입니다(경고 기준 >1px). 결과 해석에 주의하세요."
                elif message.startswith("A/B sharpness"):
                    message=f"A/B 선명도 차이가 큽니다(Laplacian 분산비 {checks['sharpness_ratio']:.1f}배; 기준 2배)."
                elif message.startswith("GSD differs"):
                    message=f"GSD {gsd:g}m/px는 학습 기준 0.5m/px와 20% 넘게 다릅니다."
                elif message.startswith("Estimated isolated-object"):
                    message=f"고립 객체의 추정 검출 한계 ≈ {209*gsd**2:.1f}m² (pixel-fixed 가정; 0.5m에서만 검증)."
                elif message.startswith("A/B dimensions differ"):
                    message="A/B 크기가 달라 정합·선명도 검사를 할 수 없습니다."
            st.warning(message)
        if checks["shift_px"] is not None:
            st.caption(t(f"Estimated phase-correlation shift: dx={checks['shift_xy_px'][0]:.2f}px, dy={checks['shift_xy_px'][1]:.2f}px · sharpness variance ratio {checks['sharpness_ratio']:.2f}×",f"phaseCorrelate 추정 이동: dx={checks['shift_xy_px'][0]:.2f}px, dy={checks['shift_xy_px'][1]:.2f}px · 선명도 분산비 {checks['sharpness_ratio']:.2f}×"))
        probability=None
        analysis_failed=False
        appeared_mask=None; possibly_disappeared_mask=None
        if method == "TinyCD":
            try:
                weights_path=robust_weights if model_variant.startswith("ROBUST-v2") else ROOT/"models"/"tinycd_levir_best.pth"
                model,device=cached_tinycd(str(weights_path))
                probability=cached_probability(model,str(weights_path),before,after,str(device))
                if bidirectional:
                    reverse_probability=cached_probability(model,str(weights_path),after,before,str(device))
                    appeared_mask=probability>=0.5
                    possibly_disappeared_mask=reverse_probability>=0.5
                    mask=appeared_mask|possibly_disappeared_mask
                    probability=np.maximum(probability,reverse_probability)
                    st.warning(t("EXPERIMENTAL: A→B is shown as appeared; B→A as possibly disappeared. LEVIR labels have no direction, so disappearance/demolition is unvalidated.","EXPERIMENTAL: A→B는 appeared, B→A는 possibly disappeared로 표시합니다. LEVIR 정답에 방향 라벨이 없어 소멸/철거 탐지는 검증되지 않았습니다."))
                else:
                    mask=probability>=0.5
                st.caption(t(f"TinyCD device: {device} · 256px tiles · cutoff 0.5",f"TinyCD 장치: {device} · 256px tiles · cutoff 0.5"))
            except Exception as exc:
                st.error(t("TinyCD inference failed; Results and Limitations remain available.",f"TinyCD 실행 오류: {exc}"))
                analysis_failed=True
        else:
            pred,strength=opencv_predictions(before,after)[method]
            mask=pred.astype(bool)
            probability=strength
            st.caption(t("Fixed OpenCV settings: basic absdiff threshold=105 · improved Gaussian 3×3 + Otsu + 5×5 morphology","OpenCV 고정 설정: 기본 absdiff threshold=105 · 개선형 Gaussian 3×3 + Otsu + morphology 5×5"))
        if not analysis_failed:
            mask,object_filter=filter_small_components(mask,gsd,minimum_object_area_m2)
            if appeared_mask is not None:
                appeared_mask=appeared_mask & mask
                possibly_disappeared_mask=possibly_disappeared_mask & mask
            summary=changed_object_summary(mask,gsd)
            cols=st.columns(4)
            cols[0].metric(t("Changed objects","변화 객체"),f"{summary['count']:,}")
            cols[1].metric(t("Total area","변화 면적"),f"{summary['total_area_m2']:,.1f} m²")
            cols[2].metric(t("Changed pixels","변화 픽셀"),f"{summary['changed_pixels']:,}")
            cols[3].metric(t("Median object area","객체 중앙 면적"),f"{summary['median_object_area_m2']:,.1f} m²")
            filter_message=(
                f"Object-area filter: ≥{object_filter['minimum_area_m2']:,.2f} m² "
                f"(≥{object_filter['minimum_pixels_at_gsd']} px at {gsd:.2f} m/px). "
                f"Removed {object_filter['removed_sub_floor_components']:,} components below the area floor "
                f"({object_filter['raw_count']:,} → {object_filter['kept_count']:,} objects)."
            )
            if minimum_object_area_m2 == 0:
                filter_message=(
                    f"Object-area filter is off. Removed {object_filter['removed_sub_floor_components']:,} components "
                    f"({object_filter['raw_count']:,} → {object_filter['kept_count']:,} objects)."
                )
            if LANG=="한국어":
                if minimum_object_area_m2 == 0:
                    filter_message=(f"객체 면적 필터 꺼짐. component {object_filter['removed_sub_floor_components']:,}개 제외 "
                                    f"({object_filter['raw_count']:,} → {object_filter['kept_count']:,}개).")
                else:
                    filter_message=(f"객체 면적 필터: ≥{object_filter['minimum_area_m2']:,.2f}m² "
                                    f"(현재 GSD에서 ≥{object_filter['minimum_pixels_at_gsd']}px). "
                                    f"면적 기준 미만 component {object_filter['removed_sub_floor_components']:,}개 제외 "
                                    f"({object_filter['raw_count']:,} → {object_filter['kept_count']:,}개).")
            st.caption(filter_message)
            images=[before,after,probability_heatmap(probability),make_overlay(after,mask)]
            captions=["A = before (older)", "B = after (newer)", t("Change probability / intensity heatmap","변화 확률/강도 heatmap"),t("Change mask overlay (experimental union)","변화 mask overlay (실험 union)") if bidirectional else t("Change mask overlay","변화 mask overlay")]
            panels=st.columns(4)
            for col,caption,image in zip(panels,captions,images):
                col.image(image,caption=caption,width="stretch")
            panels[2].image(probability_colorbar(),width="stretch")
            legend_labels=panels[2].columns(3)
            legend_labels[0].caption(t("0.0 low","0.0 낮음"))
            legend_labels[1].caption("0.5")
            legend_labels[2].caption(t("1.0 high","1.0 높음"))
            panels[2].caption(t("TinyCD probability / normalized OpenCV change intensity","TinyCD 확률 / OpenCV 정규화 변화 강도"))
            st.subheader(t("Before / after swipe comparison","Before / after swipe 비교"))
            slider_scale=min(1.0,1200/max(before.shape[1],after.shape[1]))
            if slider_scale<1:
                slider_size=(round(before.shape[1]*slider_scale),round(before.shape[0]*slider_scale))
                slider_a=np.asarray(Image.fromarray(before).resize(slider_size,Image.Resampling.LANCZOS));slider_b=np.asarray(Image.fromarray(after).resize(slider_size,Image.Resampling.LANCZOS))
            else: slider_a,slider_b=before,after
            components.html(comparison_slider(slider_a,slider_b),height=525,scrolling=False)
            object_rows,object_geojson=object_features(mask,probability,geo,gsd)
            st.subheader(t("Changed objects","변화 객체 목록"))
            st.caption(t("Sort by area, centroid, or mean probability. Georeferenced inputs use a local UTM CRS for area and geographic centroids.","표를 정렬해 확인하세요. 지오참조 입력은 지역 UTM 좌표계로 면적을 계산하고 중심 좌표를 경위도로 표시합니다."))
            st.dataframe(object_rows,hide_index=True,width="stretch")
            csv_buffer=BytesIO()
            columns=["object_id","area_m2","centroid_lon","centroid_lat","centroid_x_px","centroid_y_px","mean_probability","pixel_count","area_crs"]
            text_buffer=__import__("io").StringIO()
            writer=csv.DictWriter(text_buffer,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(object_rows)
            csv_buffer.write(text_buffer.getvalue().encode("utf-8-sig"));csv_buffer.seek(0)
            d1,d2=st.columns(2)
            d1.download_button(t("Download objects CSV","객체 CSV 다운로드"),data=csv_buffer.getvalue(),file_name="astra_change_objects.csv",mime="text/csv",key="download_objects_csv")
            if geo and object_geojson:
                geojson_bytes=json.dumps(object_geojson,ensure_ascii=False).encode("utf-8")
                d2.download_button(t("Download GeoJSON","GeoJSON 다운로드"),data=geojson_bytes,file_name="astra_change_objects.geojson",mime="application/geo+json",key="download_objects_geojson")
                st.subheader(t("Change object map","변화 객체 지도"))
                if object_rows:
                    center=[float(np.mean([r["centroid_lat"] for r in object_rows])),float(np.mean([r["centroid_lon"] for r in object_rows]))]
                    fmap=folium.Map(location=center,zoom_start=15,control_scale=True,tiles="OpenStreetMap")
                    if geo:
                        from rasterio.warp import transform as transform_coords
                        h,w=after.shape[:2]; affine=geo["transform"]
                        corners=[affine*(0,0),affine*(w,0),affine*(w,h),affine*(0,h)]
                        lons,lats=transform_coords(geo["crs"],"EPSG:4326",[p[0] for p in corners],[p[1] for p in corners])
                        bounds=[[min(lats),min(lons)],[max(lats),max(lons)]]
                        folium.raster_layers.ImageOverlay(after,bounds=bounds,opacity=.78,name="B · after raster",interactive=True,zindex=1).add_to(fmap)
                        rgba=np.zeros((*mask.shape,4),dtype=np.uint8);rgba[mask]=[255,48,48,150]
                        folium.raster_layers.ImageOverlay(rgba,bounds=bounds,opacity=1.0,name="Change mask raster",interactive=True,zindex=2).add_to(fmap)
                    folium.GeoJson(object_geojson,name="Change objects",
                        tooltip=folium.GeoJsonTooltip(fields=["object_id","area_m2","mean_probability"],aliases=["ID","Area m²","Mean probability"]),
                        style_function=lambda _: {"color":"#ef4444","weight":2,"fillColor":"#f97316","fillOpacity":0.35}).add_to(fmap)
                    folium.LayerControl().add_to(fmap)
                    if SCREENSHOT_HIDE_MAP:
                        st.caption(t("Interactive map hidden for headless screenshot capture only.","헤드리스 스크린샷 캡처에서만 지도를 숨겼습니다."))
                    else:
                        st_folium(fmap,height=520,width=None,key="change_object_map")
                else:
                    st.info(t("No changed objects were found in the current mask.","현재 mask에서 변화 객체를 찾지 못했습니다."))
            elif geo is None:
                d2.caption(t("GeoJSON and geospatial mapping require georeferenced GeoTIFF inputs.","GeoJSON/지도는 지오참조 GeoTIFF 입력에서 사용할 수 있습니다."))
            if bidirectional and appeared_mask is not None:
                left,right=st.columns(2)
                left.image(make_overlay(after,appeared_mask,(0,220,90)),caption="Appeared · A→B (green)",width="stretch")
                right.image(make_overlay(after,possibly_disappeared_mask,(235,35,180)),caption="Possibly disappeared · B→A (magenta)",width="stretch")

with tab_results:
    render_results()

with tab_limits:
    st.subheader(t("Scope and limitations","적용 범위와 한계"))
    st.markdown(t("""
- **Domain:** Quantitative results cover LEVIR-CD only; generalization to other sensors or regions is untested.
- **Target:** The benchmark labels building changes. Do not treat this as a detector for roads, vegetation, water, or arbitrary change.
- **Direction:** TinyCD behaves like an appearance detector, likely reflecting growth-heavy LEVIR labels. Disappearance/demolition is unvalidated.
- **Small objects:** Isolated changes below about 50 m² are often missed.
- **Use:** Research and education demo only; not for operational decisions.
- **Registration:** A/B shifts above 1 px raise a warning; phase correlation can also react to real changes.
- **Resolution:** Trained at about 0.5 m/px and sensitive to coarser GSD, sharpness mismatch, and blur.
- **Licensing:** Follow LEVIR academic terms and TinyCD non-commercial research terms. Data, weights, and image previews are excluded from Git.
- **NAIP:** Small public-domain USDA imagery window, qualitative only, with no ground truth.
- **GeoTIFF:** GeoJSON and map coordinates require georeferenced GeoTIFF inputs.
- **Safety:** Research/education demo; never use for real operational decisions.
""", """
- **도메인:** 정량 벤치마크는 LEVIR-CD에 한정되며 다른 센서·지역 일반화는 미검증입니다.
- **대상:** 건물 변화 라벨 기준입니다. 도로·식생·수면 등 임의 변화 탐지기로 해석하지 마세요.
- **방향성:** TinyCD는 appearance detector처럼 보이며 성장 위주인 LEVIR 라벨을 반영했을 수 있습니다. 소멸/철거는 검증되지 않았습니다.
- **작은 객체:** 고립 변화 중 약 50m² 미만은 자주 누락됩니다.
- **사용:** 연구·교육용 데모이며 실제 운용 판단에 사용할 수 없습니다.
- **정합:** A/B 1px 초과 이동 시 경고합니다. phase correlation은 실제 변화에도 영향을 받을 수 있습니다.
- **해상도:** 약 0.5m/px에서 학습했으며 더 거친 GSD, 선명도 불일치, blur에 민감합니다.
- **라이선스:** LEVIR 학술 조건과 TinyCD 비상업 연구 조건을 따릅니다. 데이터·가중치·미리보기는 Git에서 제외합니다.
- **NAIP:** USDA 공개 도메인 항공영상의 작은 영역으로, 정답이 없는 질적 결과입니다.
- **GeoTIFF:** GeoJSON과 지도 좌표는 지오참조 GeoTIFF에서만 제공합니다.
- **안전:** 연구·교육용 데모이며 실제 운용 판단에 사용할 수 없습니다.
"""))
