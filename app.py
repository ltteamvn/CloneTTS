# -*- coding: utf-8 -*-
"""
Chuyển Giọng Nói AI — Bản nâng cấp toàn diện
- Đọc SRT chuẩn timing, tự điều tốc theo từng câu để khớp cửa sổ thời gian
- Không chồng/thiếu giữa các đoạn
- Tự nhận GPU (CUDA/MPS) nếu có, fallback CPU
- Giao diện gọn đẹp và chuyên nghiệp hơn (Gradio + Theme + CSS)
"""

import os
import sys
import json
import math
import time
import asyncio
import threading
from datetime import datetime

import torch
import gradio as gr
import pydub
from pydub.effects import speedup as pydub_speedup
import edge_tts
import srt

# ──────────────────────────────────────────────────────────────────────────────
# 1) Bảo đảm import được ChatterboxVC (src/)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(SCRIPT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import importlib
import chatterbox.vc
importlib.reload(chatterbox.vc)
from chatterbox.vc import ChatterboxVC

# ──────────────────────────────────────────────────────────────────────────────
# 2) Thiết bị: Tự nhận GPU (CUDA/MPS) nếu có, ngược lại dùng CPU
# ──────────────────────────────────────────────────────────────────────────────
def detect_device():
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA"
        return "cuda", gpu_name
    # Hỗ trợ MPS (Apple Silicon) nếu muốn; nếu model không hỗ trợ, sẽ fallback CPU.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple MPS"
    return "cpu", None

DEVICE, GPU_NAME = detect_device()

_vc_model = None
def get_vc_model():
    """Lazy load VC model theo DEVICE đã phát hiện."""
    global _vc_model
    if _vc_model is None:
        print(f"[VC] Đang tải model trên {DEVICE}{' ('+GPU_NAME+')' if GPU_NAME else ''}…")
        try:
            _vc_model = ChatterboxVC.from_pretrained(DEVICE)
        except Exception as e:
            # Fallback an toàn nếu MPS không được hỗ trợ
            if DEVICE == "mps":
                print("[VC] MPS không được hỗ trợ. Chuyển sang CPU.")
                _vc_model = ChatterboxVC.from_pretrained("cpu")
            else:
                raise e
        print("[VC] Model sẵn sàng.")
    return _vc_model

# ──────────────────────────────────────────────────────────────────────────────
# 3) UI log helper (Gradio)
# ──────────────────────────────────────────────────────────────────────────────
global_log_messages_vc = []
def yield_vc_updates(log_msg=None, audio_data=None, file_list=None, log_append=True):
    """Cập nhật log/âm thanh/files ra UI (dùng với generator)."""
    global global_log_messages_vc
    # cập nhật log
    if log_msg is not None:
        prefix = datetime.now().strftime("[%H:%M:%S]")
        if log_append:
            global_log_messages_vc.append(f"{prefix} {log_msg}")
        else:
            global_log_messages_vc = [f"{prefix} {log_msg}"]
    log_update = gr.update(value="\n".join(global_log_messages_vc))

    # audio output
    audio_update = gr.update(visible=(audio_data is not None),
                             value=audio_data if audio_data is not None else None)
    # file-download output
    files_update = gr.update(visible=(file_list is not None),
                             value=file_list if file_list is not None else [])

    yield log_update, audio_update, files_update

# ──────────────────────────────────────────────────────────────────────────────
# 4) Tải danh sách giọng Edge TTS từ voices.json
# ──────────────────────────────────────────────────────────────────────────────
def load_edge_tts_voices(json_path="voices.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        voices = json.load(f)
    display_list, code_map = [], {}
    for lang, genders in voices.items():
        for gender, items in genders.items():
            for v in items:
                disp = f"{lang} - {gender} - {v['display_name']} ({v['voice_code']})"
                display_list.append(disp)
                code_map[disp] = v["voice_code"]
    display_list.sort()
    return display_list, code_map

EDGE_CHOICES, EDGE_CODE_MAP = load_edge_tts_voices()

# ──────────────────────────────────────────────────────────────────────────────
# 5) Edge TTS helpers (an toàn trong môi trường có/không event loop)
# ──────────────────────────────────────────────────────────────────────────────
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _norm_srt_text(txt: str) -> str:
    # Hợp nhất dòng, loại khoảng trắng thừa
    t = " ".join(str(txt).replace("\n", " ").split())
    return t.strip()

async def _edge_tts_async(text, voice_disp, rate_pct, vol_pct, out_path):
    code = EDGE_CODE_MAP.get(voice_disp)
    if not code:
        raise ValueError("Không tìm thấy voice trong voices.json.")
    rate_str = f"{int(rate_pct):+d}%"
    vol_str  = f"{int(vol_pct):+d}%"
    comm = edge_tts.Communicate(text, voice=code, rate=rate_str, volume=vol_str)
    await comm.save(out_path)
    return out_path

def run_edge_tts_sync(text, voice_disp, rate_pct, vol_pct, out_path):
    """
    Chạy edge-tts đồng bộ, an toàn dù trong/ngoài event loop (Gradio).
    """
    result = {"path": None, "err": None}

    def _runner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            p = loop.run_until_complete(_edge_tts_async(text, voice_disp, rate_pct, vol_pct, out_path))
            result["path"] = p
        except Exception as e:
            result["err"] = e
        finally:
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if result["err"] is not None:
        raise result["err"]
    return result["path"]

# ──────────────────────────────────────────────────────────────────────────────
# 6) SRT → Audio: tự điều tốc để khớp cửa sổ thời gian từng đoạn
# ──────────────────────────────────────────────────────────────────────────────
def _fit_tts_to_window(text, voice_disp, base_rate_pct, vol_pct, target_ms, tmp_dir, idx,
                       tol_ratio=0.08, max_trials=3):
    """
    Sinh TTS cho 1 câu và cố gắng có độ dài khớp target_ms:
    - Thử đọc ở tốc độ base_rate_pct, đo độ dài
    - Nếu quá dài so với khung, tăng rate tương ứng để đẩy nhanh
    - Nếu ngắn, giữ nguyên (sau sẽ pad im lặng). Mục tiêu: không cắt chữ.
    - Nếu vẫn lệch sau vài lần, dùng speedup (pydub) nhẹ để tinh chỉnh (chỉ speed-up).
    """
    text = _norm_srt_text(text)
    if target_ms <= 0:
        target_ms = 1  # tránh chia 0

    base_factor = 1.0 + (base_rate_pct / 100.0)
    used_rate = base_rate_pct
    wav_path = os.path.join(tmp_dir, f"seg_{idx:04d}.wav")

    # Lần 1: tốc độ gốc
    run_edge_tts_sync(text, voice_disp, used_rate, vol_pct, wav_path)
    seg = pydub.AudioSegment.from_file(wav_path)
    L = len(seg)

    # Nếu đã khớp trong sai số cho phép, trả về
    if abs(L - target_ms) / float(target_ms) <= tol_ratio:
        return seg, used_rate

    # Nếu dài hơn nhiều, tăng rate để rút gọn
    trials = 1
    while L > target_ms and trials < max_trials:
        need_factor = L / float(target_ms)  # cần nhanh hơn bấy nhiêu lần
        # Tốc độ mới ~ base_factor * need_factor
        new_factor = (1.0 + used_rate / 100.0) * need_factor
        new_rate = int(round((new_factor - 1.0) * 100))
        # Giới hạn để tránh quá đà
        new_rate = _clamp(new_rate, -50, 100)
        if new_rate == used_rate:
            new_rate = _clamp(used_rate + 5, -50, 100)
        used_rate = new_rate

        run_edge_tts_sync(text, voice_disp, used_rate, vol_pct, wav_path)
        seg = pydub.AudioSegment.from_file(wav_path)
        L = len(seg)
        trials += 1

    # Nếu còn dài hơn target một chút, tinh chỉnh speedup (pitch có thể thay đổi nhẹ)
    if L > target_ms:
        factor = L / float(target_ms)
        # pydub_speedup chỉ rút ngắn (factor>1). Dùng 1.0 nếu sai số rất nhỏ.
        if factor > 1.02:
            # pydub_speedup tốc độ >1 làm ngắn lại
            seg = pydub_speedup(seg, playback_speed=factor, chunk_size=50, crossfade=10)
            L = len(seg)

    # Sau cùng: cắt/pad để đúng target_ms (cắt chỉ còn 1-2ms sai số, nội dung đã tăng tốc phù hợp)
    if L > target_ms:
        seg = seg[:target_ms]
    elif L < target_ms:
        seg = seg + pydub.AudioSegment.silent(duration=target_ms - L)

    return seg, used_rate

def synthesize_srt_audio_precise(srt_path: str, voice_disp: str, work_dir: str,
                                 base_rate_pct: int, vol_pct: int) -> str:
    """
    Tạo 1 file WAV từ SRT:
    - Tôn trọng mốc thời gian start/end của từng câu
    - Mỗi câu tự điều tốc để không bị thò ra khỏi end
    - Đảm bảo toàn timeline không chồng/thiếu
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    subs.sort(key=lambda x: (x.start, x.end))

    combined = pydub.AudioSegment.empty()
    current_ms = 0
    tmp_dir = os.path.join(work_dir, "_tmp_srt")
    os.makedirs(tmp_dir, exist_ok=True)

    for i, sub in enumerate(subs, 1):
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms   = int(sub.end.total_seconds()   * 1000)
        if end_ms <= start_ms:
            # skip đoạn lỗi thời gian
            continue
        dur_ms   = end_ms - start_ms
        text = _norm_srt_text(sub.content)

        # Thêm im lặng cho đến thời điểm bắt đầu đoạn
        if start_ms > current_ms:
            combined += pydub.AudioSegment.silent(duration=start_ms - current_ms)
            current_ms = start_ms
        else:
            # Nếu SRT có overlap nhưng ta đã đảm bảo mỗi seg sẽ fit <= (end-start) nên
            # timeline sẽ không bị chồng khi ghép đúng theo SRT.
            pass

        seg, used_rate = _fit_tts_to_window(
            text=text,
            voice_disp=voice_disp,
            base_rate_pct=base_rate_pct,
            vol_pct=vol_pct,
            target_ms=dur_ms,
            tmp_dir=tmp_dir,
            idx=i
        )

        combined += seg
        current_ms = end_ms

    out_path = os.path.join(work_dir, "srt_source.wav")
    combined.export(out_path, format="wav")
    # Dọn dẹp tạm (có thể giữ lại để debug nếu muốn)
    try:
        for fn in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, fn))
            except Exception:
                pass
        os.rmdir(tmp_dir)
    except Exception:
        pass
    return out_path

# ──────────────────────────────────────────────────────────────────────────────
# 7) Voice Conversion (chuyển giọng)
# ──────────────────────────────────────────────────────────────────────────────
def generate_vc(
    source_audio_path,
    target_voice_path,
    cfg_rate: float,
    sigma_min: float,
    batch_mode: bool,
    batch_parameter: str,
    batch_values: str
):
    model = get_vc_model()
    yield from yield_vc_updates("Khởi tạo chuyển giọng…", log_append=False)

    date_folder = datetime.now().strftime("%Y%m%d")
    work_dir = os.path.join("outputs", "vc", date_folder)
    os.makedirs(work_dir, exist_ok=True)

    def run_once(src, tgt, rate, sigma):
        return model.generate(src, target_voice_path=tgt, inference_cfg_rate=rate, sigma_min=sigma)

    outputs = []
    try:
        if batch_mode:
            try:
                vals = [float(v.strip()) for v in batch_values.split(",") if v.strip()]
            except:
                raise gr.Error("Batch values phải là số, phân cách bởi dấu phẩy.")
            yield from yield_vc_updates(f"Chạy batch '{batch_parameter}': {vals}")
            for idx, v in enumerate(vals, 1):
                r, s = cfg_rate, sigma_min
                tag = ""
                if batch_parameter == "Inference CFG Rate":
                    r, tag = v, f"cfg_{v}"
                else:
                    s, tag = v, f"sigma_{v}"
                yield from yield_vc_updates(f" • Mục {idx}/{len(vals)}: {batch_parameter}={v}")
                wav = run_once(source_audio_path, target_voice_path, r, s)
                fn = f"{tag}_{idx}.wav"
                path = os.path.join(work_dir, fn)
                model.save_wav(wav, path)
                outputs.append(path)
                yield from yield_vc_updates(f"Đã lưu: {path}")
        else:
            audio = pydub.AudioSegment.from_file(source_audio_path)
            if len(audio) > 40_000:
                yield from yield_vc_updates("Audio dài >40s: tách thành đoạn 40s…")
                chunks = [audio[i:i+40_000] for i in range(0, len(audio), 40_000)]
                temp_paths = []
                for i, chunk in enumerate(chunks):
                    tmp = f"{source_audio_path}_chunk{i}.wav"
                    chunk.export(tmp, format="wav")
                    wav = run_once(tmp, target_voice_path, cfg_rate, sigma_min)
                    outp = os.path.join(work_dir, f"part{i}.wav")
                    model.save_wav(wav, outp)
                    temp_paths.append(outp)
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                    yield from yield_vc_updates(f"Xử lý đoạn {i+1}/{len(chunks)}")
                # ghép lại
                combined = pydub.AudioSegment.empty()
                for p in temp_paths:
                    combined += pydub.AudioSegment.from_file(p)
                final = os.path.join(work_dir, "combined.wav")
                combined.export(final, format="wav")
                outputs.append(final)
                yield from yield_vc_updates("Chuyển xong.")
            else:
                yield from yield_vc_updates("Đang chuyển giọng…")
                wav = run_once(source_audio_path, target_voice_path, cfg_rate, sigma_min)
                outp = os.path.join(work_dir, f"LyTranTTS_{datetime.now().strftime('%H%M%S')}.wav")
                model.save_wav(wav, outp)
                outputs.append(outp)
                yield from yield_vc_updates("Hoàn thành.")
    except Exception as e:
        yield from yield_vc_updates(f"Lỗi: {e}")
        raise

    # Trả về audio đầu tiên và danh sách file kết quả
    first = outputs[0] if outputs else None
    yield from yield_vc_updates(log_msg=None, audio_data=first, file_list=outputs)

# ──────────────────────────────────────────────────────────────────────────────
# 8) Wrapper: chọn nguồn (SRT / Edge TTS / File) → VC
# ──────────────────────────────────────────────────────────────────────────────
def run_vc_pipeline(
    source_mode,                # "SRT" | "Edge TTS" | "Tệp/Ghi âm"
    srt_file, srt_voice, srt_rate, srt_vol,
    edge_text, edge_voice, edge_rate, edge_vol,
    src_audio, tgt_audio,
    cfg_rate, sigma_min,
    batch_mode, batch_parameter, batch_values
):
    # Reset log đầu phiên
    yield from yield_vc_updates(f"Bắt đầu trên thiết bị: {DEVICE.upper()}{' - '+GPU_NAME if GPU_NAME else ''}", log_append=False)

    date_folder = datetime.now().strftime("%Y%m%d")
    work_dir = os.path.join("outputs", "vc", date_folder)
    os.makedirs(work_dir, exist_ok=True)

    # 1) Chuẩn bị nguồn
    if source_mode == "SRT":
        if not srt_file:
            raise gr.Error("Hãy tải lên file .srt")
        if not srt_voice:
            raise gr.Error("Hãy chọn giọng Edge TTS cho SRT.")
        yield from yield_vc_updates("Đang tổng hợp nguồn từ SRT (canh mốc thời gian, tự điều tốc)…")
        source = synthesize_srt_audio_precise(
            srt_path=srt_file.name,
            voice_disp=srt_voice,
            work_dir=work_dir,
            base_rate_pct=int(srt_rate),
            vol_pct=int(srt_vol),
        )
    elif source_mode == "Edge TTS":
        if not edge_text or not edge_voice:
            raise gr.Error("Hãy nhập văn bản và chọn giọng cho Edge TTS.")
        yield from yield_vc_updates("Đang tạo nguồn từ Edge TTS…")
        tmp_path = os.path.join(work_dir, "edge_source.wav")
        run_edge_tts_sync(edge_text, edge_voice, int(edge_rate), int(edge_vol), tmp_path)
        source = tmp_path
    else:  # "Tệp/Ghi âm"
        if not src_audio:
            raise gr.Error("Hãy tải lên hoặc ghi âm nguồn.")
        source = src_audio

    # 2) Voice Conversion
    if not tgt_audio:
        raise gr.Error("Hãy tải lên/ghi âm giọng mục tiêu (tham chiếu).")

    yield from generate_vc(
        source_audio_path=source,
        target_voice_path=tgt_audio,
        cfg_rate=cfg_rate,
        sigma_min=sigma_min,
        batch_mode=batch_mode,
        batch_parameter=batch_parameter,
        batch_values=batch_values
    )

# ──────────────────────────────────────────────────────────────────────────────
# 9) Giao diện Gradio — gọn đẹp và chuyên nghiệp
# ──────────────────────────────────────────────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
)

CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; }
.header-title { display:flex; align-items:center; gap:.75rem; }
.badge { padding:.15rem .5rem; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:600; font-size:.85rem; }
.kv { display:flex; gap:.5rem; flex-wrap:wrap; font-size:.9rem; color:#334155; }
.footer-note { color:#64748b; font-size:.85rem; }
"""

with gr.Blocks(title="Chuyển Giọng Nói AI — Pro", theme=THEME, css=CUSTOM_CSS) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"""
<div class="header-title">
  <h2>🎙️ Chuyển Giọng Nói AI — Pro</h2>
  <span class="badge">{'GPU: ' + GPU_NAME if GPU_NAME else ('CPU' if DEVICE=='cpu' else DEVICE.upper())}</span>
</div>
<div class="kv">
  <div><strong>Thiết bị:</strong> {DEVICE.upper()}</div>
  <div><strong>Ngày:</strong> {datetime.now().strftime('%d/%m/%Y')}</div>
</div>
""",
                elem_id="header"
            )
        with gr.Column(scale=0.3, min_width=220):
            clear_btn = gr.Button("🧹 Xóa nhật ký", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1, min_width=420):
            gr.Markdown("### 1) Nguồn âm thanh")
            source_mode = gr.Dropdown(
                label="Chọn nguồn",
                choices=["Tệp/Ghi âm", "SRT", "Edge TTS"],
                value="Tệp/Ghi âm"
            )

            # --- Nhóm SRT ---
            with gr.Group(visible=False) as grp_srt:
                srt_file  = gr.File(file_types=[".srt"], label="Tải lên file .srt")
                srt_voice = gr.Dropdown(choices=EDGE_CHOICES, label="Giọng Edge TTS (đọc SRT)")
                srt_rate  = gr.Slider(-100, 100, value=0, step=1, label="Tốc độ nền SRT (% chuẩn)")
                srt_vol   = gr.Slider(-100, 100, value=0, step=1, label="Âm lượng SRT (% chuẩn)")

            # --- Nhóm Edge TTS ---
            with gr.Group(visible=False) as grp_edge:
                edge_text  = gr.Textbox(label="Văn bản cho Edge TTS", lines=4, placeholder="Nhập nội dung…")
                edge_voice = gr.Dropdown(choices=EDGE_CHOICES, label="Giọng Edge TTS")
                edge_rate  = gr.Slider(-100, 100, value=0, step=1, label="Tốc độ Edge (% chuẩn)")
                edge_vol   = gr.Slider(-100, 100, value=0, step=1, label="Âm lượng Edge (% chuẩn)")
                gen_edge_btn = gr.Button("🗣️ Tạo Edge TTS (xem/ghi làm nguồn)")
                edge_audio   = gr.Audio(label="Nguồn Edge TTS", type="filepath", visible=True)

            # --- Nhóm File/Ghi âm ---
            with gr.Group(visible=True) as grp_file:
                src_audio = gr.Audio(sources=["upload","microphone"], type="filepath",
                                     label="Tải lên / Ghi âm nguồn")

            gr.Markdown("### 2) Giọng mục tiêu (tham chiếu)")
            tgt_audio = gr.Audio(sources=["upload","microphone"], type="filepath",
                                 label="Tải lên / Ghi âm giọng mục tiêu")

            gr.Markdown("### 3) Tham số chuyển giọng")
            cfg_slider  = gr.Slider(0.0, 30.0, value=0.5, step=0.1, label="Inference CFG Rate")
            sigma_input = gr.Number(value=1e-6, label="Sigma Min",
                                    minimum=1e-7, maximum=1e-5, precision=8)

            with gr.Accordion("Tùy chọn Batch Sweep (nâng cao)", open=False):
                batch_chk   = gr.Checkbox(label="Kích hoạt Batch Sweep", value=False)
                batch_param = gr.Dropdown(choices=["Inference CFG Rate","Sigma Min"],
                                          label="Tham số thay đổi")
                batch_vals  = gr.Textbox(placeholder="ví dụ: 0.5,1.0,2.0",
                                         label="Giá trị (phân cách dấu phẩy)")

            run_btn = gr.Button("🚀 Bắt đầu chuyển giọng", variant="primary")

        with gr.Column(scale=1, min_width=420):
            gr.Markdown("### Nhật ký")
            log_box = gr.Textbox(interactive=False, lines=16)
            gr.Markdown("### Kết quả")
            out_audio = gr.Audio(label="Âm thanh kết quả", type="filepath", visible=False)
            out_files = gr.Files(label="Tải xuống các tệp đầu ra", visible=False)
            gr.Markdown('<div class="footer-note">* Hệ thống tự giữ nguyên nội dung câu, chỉ thay đổi tốc độ để khớp khung thời gian SRT.</div>')

    # ── Toggle hiển thị nhóm theo nguồn ───────────────────────────────────────
    def _toggle_groups(mode):
        return (
            gr.update(visible=(mode == "SRT")),
            gr.update(visible=(mode == "Edge TTS")),
            gr.update(visible=(mode == "Tệp/Ghi âm")),
        )

    source_mode.change(
        fn=_toggle_groups,
        inputs=[source_mode],
        outputs=[grp_srt, grp_edge, grp_file]
    )

    # ── Xóa log ───────────────────────────────────────────────────────────────
    def clear_logs():
        global global_log_messages_vc
        global_log_messages_vc = []
        return gr.update(value="")

    clear_btn.click(fn=clear_logs, inputs=None, outputs=[log_box])

    # ── Sinh Edge TTS preview/nguồn ───────────────────────────────────────────
    def _gen_edge_preview(text, voice, rate, vol):
        if not text or not voice:
            raise gr.Error("Hãy nhập văn bản và chọn giọng Edge TTS.")
        date_folder = datetime.now().strftime("%Y%m%d")
        work_dir = os.path.join("outputs", "vc", date_folder)
        os.makedirs(work_dir, exist_ok=True)
        tmp = os.path.join(work_dir, "edge_preview.wav")
        run_edge_tts_sync(text, voice, int(rate), int(vol), tmp)
        return tmp, tmp

    gen_edge_btn.click(
        fn=_gen_edge_preview,
        inputs=[edge_text, edge_voice, edge_rate, edge_vol],
        outputs=[edge_audio, src_audio]
    )

    # ── Chạy pipeline VC ──────────────────────────────────────────────────────
    run_btn.click(
        fn=run_vc_pipeline,
        inputs=[
            source_mode,
            # SRT
            srt_file, srt_voice, srt_rate, srt_vol,
            # Edge
            edge_text, edge_voice, edge_rate, edge_vol,
            # File
            src_audio, tgt_audio,
            # VC params
            cfg_slider, sigma_input,
            batch_chk, batch_param, batch_vals
        ],
        outputs=[log_box, out_audio, out_files],
        show_progress="minimal"
    )

if __name__ == "__main__":
    # Bạn có thể tắt share nếu không cần: demo.launch(share=False)
    demo.launch(share=True)
