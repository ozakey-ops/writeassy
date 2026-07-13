"""
글 첨삭 도우미 Pro
Google Cloud Vision API (OCR) + Gemini AI (첨삭)

API 키는 소스코드에 절대 포함되지 않습니다.
로드 우선순위: st.secrets → 환경변수(os.environ)

로컬 실행:
  VISION_API_KEY=AIza... GEMINI_API_KEY=AIza... streamlit run app.py
  또는 .streamlit/secrets.toml 에 키 설정

배포 (Streamlit Cloud):
  앱 Settings → Secrets 에 VISION_API_KEY / GEMINI_API_KEY 입력
"""

import os
import re
import io
import json
import time
import base64
from datetime import datetime

import streamlit as st
import requests
from PIL import Image
import google.generativeai as genai

# ── 고정 상수 ─────────────────────────────────────────────────────
GEMINI_MODEL = "​gemini-3-flash"   # 사용 모델

# ── 페이지 설정 ────────────────────────────────────────────────────
st.set_page_config(
    page_title="글 첨삭 도우미 Pro",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { max-width: 720px; padding-top: 1.5rem; padding-bottom: 3rem; }

.result-box {
    background: #eef2ff; border-left: 4px solid #4f46e5;
    border-radius: 12px; padding: 18px 20px;
    line-height: 2.1; white-space: pre-wrap;
    font-size: 14px; color: #111827; margin-bottom: 4px;
}
.orig-box {
    background: #f9fafb; border-left: 4px solid #d1d5db;
    border-radius: 12px; padding: 18px 20px;
    line-height: 2.1; white-space: pre-wrap;
    font-size: 14px; color: #6b7280;
}
.c-item {
    background: #f3f4f6; border-radius: 10px;
    padding: 11px 15px; margin-bottom: 8px;
    font-size: 13px; line-height: 1.7;
}
.c-tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700; margin-right: 8px;
}
.tag-grammar { background: #fef3c7; color: #92400e; }
.tag-style   { background: #dbeafe; color: #1e40af; }
.tag-logic   { background: #d1fae5; color: #065f46; }
.tag-flow    { background: #ede9fe; color: #5b21b6; }
.score-pill {
    display: inline-block; padding: 5px 20px; border-radius: 20px;
    font-weight: 800; font-size: 15px; color: white;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: #4f46e5; color: white;
    font-size: 13px; font-weight: 800;
    margin-right: 8px; vertical-align: middle;
}
.setup-box {
    background: #fefce8; border: 1px solid #fde047;
    border-radius: 12px; padding: 20px 22px;
    font-size: 13px; line-height: 2; color: #713f12;
}
.setup-box code {
    background: #fef9c3; padding: 2px 6px;
    border-radius: 5px; font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── API 키 로드 (소스코드에 키 없음) ─────────────────────────────
def _load_key(name: str) -> str:
    """st.secrets → 환경변수 순서로 로드. 없으면 빈 문자열."""
    try:
        val = st.secrets[name]
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(name, "")

VISION_KEY = _load_key("VISION_API_KEY")
GEMINI_KEY = _load_key("GEMINI_API_KEY")
KEYS_OK    = bool(VISION_KEY) and bool(GEMINI_KEY)


# ── 세션 상태 초기화 ──────────────────────────────────────────────
_defaults = {
    "ocr_text":      "",
    "orig_text":     "",
    "edited_text":   "",
    "criteria":      [],
    "score":         None,
    "analysis_done": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── 사이드바 (키 입력 없음 — 상태 표시만) ────────────────────────
with st.sidebar:
    st.markdown("## ✍️ 글 첨삭 도우미 Pro")
    st.caption("Google Vision OCR · Gemini AI")
    st.divider()

    if KEYS_OK:
        st.success("✅ API 연결됨")
        st.caption(f"모델: `{GEMINI_MODEL}`")
    else:
        st.error("⚠️ API 키 미설정")
        missing = []
        if not VISION_KEY: missing.append("VISION_API_KEY")
        if not GEMINI_KEY:  missing.append("GEMINI_API_KEY")
        st.caption(f"누락된 키: {', '.join(missing)}")

    st.divider()
    if st.button("🔄 처음부터 다시", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()


# ── 유틸 함수 ─────────────────────────────────────────────────────
def img_to_b64(pil_image: Image.Image, max_px: int = 2000, quality: int = 92) -> str:
    img = pil_image.copy()
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def run_ocr(image: Image.Image) -> str:
    """Google Cloud Vision API — DOCUMENT_TEXT_DETECTION"""
    b64 = img_to_b64(image)
    resp = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={VISION_KEY}",
        json={"requests": [{
            "image": {"content": b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["ko", "en"]},
        }]},
        timeout=30,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    r0 = data.get("responses", [{}])[0]
    if "error" in r0:
        raise RuntimeError(r0["error"]["message"])
    text = r0.get("fullTextAnnotation", {}).get("text", "").strip()
    if not text:
        raise RuntimeError("텍스트를 찾을 수 없습니다. 더 선명한 사진을 사용해주세요.")
    return text


def call_gemini(prompt: str) -> str:
    """Gemini 호출 — 할당량 초과 시 자동 대기 후 1회 재시도"""
    genai.configure(api_key=GEMINI_KEY)
    gm  = genai.GenerativeModel(GEMINI_MODEL)
    cfg = genai.GenerationConfig(temperature=0.3, max_output_tokens=2048)

    for attempt in range(2):
        try:
            return gm.generate_content(prompt, generation_config=cfg).text
        except Exception as e:
            msg = str(e)
            is_quota = any(k in msg for k in ("quota", "429", "RESOURCE_EXHAUSTED"))
            if is_quota and attempt == 0:
                m_sec = re.search(r"retry[^\d]*([\d.]+)\s*s", msg)
                wait  = float(m_sec.group(1)) + 3 if m_sec else 63
                st.toast(f"⏳ 할당량 초과 — {wait:.0f}초 대기 후 자동 재시도...")
                time.sleep(wait)
                continue
            if is_quota:
                raise RuntimeError("Gemini 무료 할당량이 꽉 찼습니다. 1분 후 다시 시도해주세요.")
            raise RuntimeError(msg)


def render_criteria(criteria: list):
    tag_map = {
        "grammar": ("문법/맞춤법", "tag-grammar"),
        "style":   ("문체/표현",   "tag-style"),
        "logic":   ("논리/내용",   "tag-logic"),
        "flow":    ("흐름/구성",   "tag-flow"),
    }
    if not criteria:
        st.success("✅ 큰 문제가 없습니다. 전반적으로 다듬겠습니다.")
        return
    for c in criteria:
        label, css = tag_map.get(c.get("type", "style"), (c.get("label", ""), "tag-style"))
        st.markdown(
            f'<div class="c-item">'
            f'<span class="c-tag {css}">{label}</span>{c.get("issue","")}'
            f'</div>', unsafe_allow_html=True,
        )


def esc(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def build_txt() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    crit = "\n".join(f"• [{c.get('label','')}] {c.get('issue','')}"
                     for c in st.session_state.criteria) or "없음"
    return "\n".join([
        "글 첨삭 완성본 (Google Vision + Gemini)",
        f"작성일: {now}", "─"*40, "",
        "[원문]", st.session_state.orig_text, "", "─"*40, "",
        "[첨삭 기준]", crit, "", "─"*40, "",
        "[첨삭 완성본]", st.session_state.edited_text, "",
    ])


def build_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    tag_cls = {"grammar":"grammar","style":"style","logic":"logic","flow":"flow"}
    crit_html = "".join(
        f'<li><span class="tag {tag_cls.get(c.get("type","style"),"style")}">'
        f'{c.get("label","")}</span>{esc(c.get("issue",""))}</li>'
        for c in st.session_state.criteria
    ) or "<li>없음</li>"
    css = ("body{font-family:-apple-system,sans-serif;max-width:700px;margin:40px auto;"
           "padding:20px;color:#111;line-height:1.9}"
           "h1{color:#4f46e5;border-bottom:2px solid #4f46e5;padding-bottom:8px}"
           ".date{color:#9ca3af;font-size:12px;margin-bottom:24px}"
           "h2{color:#374151;font-size:15px;margin:22px 0 10px}"
           ".box{padding:16px;border-radius:12px;white-space:pre-wrap;font-size:14px;line-height:2}"
           ".o{background:#f9fafb;border-left:3px solid #d1d5db;color:#6b7280}"
           ".e{background:#eef2ff;border-left:3px solid #4f46e5}"
           "ul{list-style:none;padding:0;display:flex;flex-direction:column;gap:8px}"
           "li{padding:10px 14px;background:#f3f4f6;border-radius:9px;font-size:13px}"
           ".tag{display:inline-block;padding:2px 9px;border-radius:20px;"
           "font-size:11px;font-weight:700;margin-right:7px}"
           ".grammar{background:#fef3c7;color:#92400e}.style{background:#dbeafe;color:#1e40af}"
           ".logic{background:#d1fae5;color:#065f46}.flow{background:#ede9fe;color:#5b21b6}")
    return (f'<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>글 첨삭 완성본</title><style>{css}</style></head><body>'
            f'<h1>✍️ 글 첨삭 완성본</h1>'
            f'<p class="date">작성일: {now} | Google Vision + Gemini</p>'
            f'<h2>📄 원문</h2><div class="box o">{esc(st.session_state.orig_text)}</div>'
            f'<h2>📋 첨삭 기준</h2><ul>{crit_html}</ul>'
            f'<h2>✨ 첨삭 완성본</h2><div class="box e">{esc(st.session_state.edited_text)}</div>'
            f'</body></html>')


# ═══════════════════════════════════════════════════════════════════
# 메인 UI
# ═══════════════════════════════════════════════════════════════════

st.markdown("## ✍️ 글 첨삭 도우미 Pro")
st.markdown("Google Vision OCR · Gemini AI 첨삭")

# API 키 미설정 시 안내 배너
if not KEYS_OK:
    st.markdown("""
<div class="setup-box">
⚠️ <strong>API 키가 설정되지 않았습니다.</strong><br>
아래 두 가지 방법 중 하나로 설정하세요.<br><br>
<strong>방법 1 — .streamlit/secrets.toml 파일 생성</strong><br>
<code>VISION_API_KEY = "AIzaSy..."</code><br>
<code>GEMINI_API_KEY = "AIzaSy..."</code><br><br>
<strong>방법 2 — 환경변수로 실행</strong><br>
<code>VISION_API_KEY=AIza... GEMINI_API_KEY=AIza... streamlit run app.py</code><br><br>
Streamlit Cloud 배포 시: 앱 Settings → <strong>Secrets</strong> 탭에 입력
</div>
""", unsafe_allow_html=True)
    st.stop()

st.divider()

# ──────────────────────────────────────────────────────────────────
# STEP 1 — 사진 업로드 & OCR
# ──────────────────────────────────────────────────────────────────
st.markdown('<span class="step-num">1</span> **사진 업로드 & 텍스트 인식**',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "사진 파일",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded)
    col_img, col_btn = st.columns([3, 2])
    with col_img:
        st.image(image, caption=uploaded.name, use_container_width=True)
    with col_btn:
        st.markdown(f"**크기:** {image.width} × {image.height}px")
        st.markdown(f"**용량:** {uploaded.size / 1024:.0f} KB")
        st.markdown("")
        if st.button("🔍 텍스트 인식 시작", type="primary", use_container_width=True):
            with st.spinner("Google Vision OCR 실행 중..."):
                try:
                    text = run_ocr(image)
                    st.session_state.ocr_text    = text
                    st.session_state.analysis_done = False
                    st.session_state.edited_text   = ""
                    st.session_state.criteria      = []
                    st.session_state.score         = None
                    st.toast(f"✅ {len(text.replace(' ','').replace(chr(10),''))}자 인식 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"OCR 오류: {e}")

st.divider()

# ──────────────────────────────────────────────────────────────────
# STEP 2 — 텍스트 확인 & AI 첨삭
# ──────────────────────────────────────────────────────────────────
st.markdown('<span class="step-num">2</span> **텍스트 확인 & AI 첨삭**',
            unsafe_allow_html=True)

if st.session_state.ocr_text:
    st.success(f"✅ OCR 완료 — "
               f"{len(st.session_state.ocr_text.replace(' ','').replace(chr(10),''))}자 인식됨")

text_input = st.text_area(
    "첨삭할 글",
    value=st.session_state.ocr_text,
    height=260,
    placeholder="여기에 텍스트를 직접 입력하거나,\n위에서 사진을 업로드하면 자동으로 채워집니다.",
    label_visibility="collapsed",
)

if text_input.strip():
    c1, c2, c3 = st.columns(3)
    c1.metric("글자수", f"{len(text_input.replace(' ','').replace(chr(10),'')):,}")
    c2.metric("어절수", f"{len(text_input.split()):,}")
    c3.metric("문장수", max(len(re.findall(r'[.!?。]', text_input)), 1))

if st.button(
    "🤖 AI 첨삭 시작하기 (Gemini)",
    type="primary",
    use_container_width=True,
    disabled=len(text_input.strip()) < 5,
):
    st.session_state.orig_text = text_input.strip()

    with st.spinner("(1/2) 첨삭 기준 분석 중..."):
        try:
            crit_raw = call_gemini(
                f"""아래 글의 문제점을 분석하여 첨삭 기준을 JSON 배열로만 출력하세요.

글:
\"\"\"
{st.session_state.orig_text}
\"\"\"

출력 형식 (순수 JSON만, 코드블록 없이):
[
  {{"type":"grammar","label":"문법/맞춤법","issue":"구체적인 문제 설명"}},
  {{"type":"style","label":"문체/표현","issue":"구체적인 문제 설명"}}
]

type: grammar(문법·맞춤법) / style(문체·표현) / logic(논리·내용) / flow(흐름·구성)
- 실제 문제만 포함 (없으면 [])  - 최대 5개  - JSON만 출력"""
            )
            cleaned = re.sub(r"```json?|```", "", crit_raw).strip()
            m = re.search(r"\[[\s\S]*\]", cleaned)
            criteria = []
            if m:
                try:
                    criteria = json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            st.session_state.criteria = criteria
        except Exception as e:
            st.error(f"기준 분석 오류: {e}")
            st.stop()

    with st.spinner("(2/2) 첨삭 완성본 작성 중..."):
        try:
            crit_str = ("\n".join(
                f"- [{c.get('label','')}] {c.get('issue','')}"
                for c in st.session_state.criteria
            ) or "- 전반적으로 자연스럽고 완성도 높게 개선")

            edit_raw = call_gemini(
                f"""전문 글쓰기 첨삭 선생님으로서 아래 글을 첨삭해주세요.

원문:
\"\"\"
{st.session_state.orig_text}
\"\"\"

첨삭 기준:
{crit_str}

규칙:
1. 원문의 핵심 내용과 의도 유지
2. 문법·맞춤법 오류 수정
3. 자연스러운 문체로 개선
4. 논리적 흐름 강화
5. 마지막 줄에 ===점수===숫자 형태로 원문 완성도 점수 표기 (예: ===점수===72)
6. 첨삭된 글만 출력 (설명 없음, 마크다운 없음)"""
            )
            score_m = re.search(r"===점수===(\d+)", edit_raw)
            st.session_state.score       = int(score_m.group(1)) if score_m else None
            st.session_state.edited_text = re.sub(r"===점수===\d+", "", edit_raw).strip()
            st.session_state.analysis_done = True
        except Exception as e:
            st.error(f"첨삭 오류: {e}")
            st.stop()

    st.rerun()

st.divider()

# ──────────────────────────────────────────────────────────────────
# STEP 3 — 결과 & 저장
# ──────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    st.markdown('<span class="step-num">3</span> **첨삭 결과**', unsafe_allow_html=True)

    if st.session_state.score is not None:
        s     = st.session_state.score
        emoji = "🏆" if s >= 90 else "⭐" if s >= 75 else "👍" if s >= 60 else "📝"
        color = "#10b981" if s >= 90 else "#4f46e5" if s >= 75 else "#f59e0b" if s >= 60 else "#6b7280"
        st.markdown(
            f'원문 완성도: <span class="score-pill" style="background:{color}">{emoji} {s}점</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    with st.expander("📋 첨삭 기준", expanded=True):
        render_criteria(st.session_state.criteria)

    tab_edit, tab_cmp = st.tabs(["✨ 첨삭본", "📄 원문 비교"])
    with tab_edit:
        st.markdown(f'<div class="result-box">{esc(st.session_state.edited_text)}</div>',
                    unsafe_allow_html=True)
    with tab_cmp:
        co, ce = st.columns(2)
        with co:
            st.markdown("**원문**")
            st.markdown(f'<div class="orig-box">{esc(st.session_state.orig_text)}</div>',
                        unsafe_allow_html=True)
        with ce:
            st.markdown("**첨삭본**")
            st.markdown(f'<div class="result-box">{esc(st.session_state.edited_text)}</div>',
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("**💾 저장**")
    fname = st.text_input("파일 이름", value="첨삭완성본", label_visibility="collapsed")
    col_t, col_h = st.columns(2)
    with col_t:
        st.download_button("📄 TXT 저장",
                           data=build_txt().encode("utf-8"),
                           file_name=f"{fname}.txt", mime="text/plain",
                           use_container_width=True)
    with col_h:
        st.download_button("🌐 HTML 저장",
                           data=build_html().encode("utf-8"),
                           file_name=f"{fname}.html", mime="text/html",
                           use_container_width=True)
