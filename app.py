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
import difflib
import concurrent.futures
from datetime import datetime

import streamlit as st
import requests
from PIL import Image
import google.generativeai as genai

# ── 고정 상수 ─────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.5-flash"

# 토큰 옵션
TOKEN_OPTIONS = [
    ("절약 · 2,048 토큰", 500,   2048),
    ("보통 · 4,096 토큰", 2000,  4096),
    ("여유 · 8,192 토큰", 99999, 8192),
]

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
    line-height: 2.1; font-size: 14px; color: #111827; margin-bottom: 4px;
    white-space: pre-wrap; word-break: break-word;
}
.markup-box {
    background: #f9fafb; border-left: 4px solid #4f46e5;
    border-radius: 12px; padding: 18px 20px;
    line-height: 2.6; font-size: 14px; color: #111827; margin-bottom: 4px;
    word-break: break-word;
}
.orig-box {
    background: #f9fafb; border-left: 4px solid #d1d5db;
    border-radius: 12px; padding: 18px 20px;
    line-height: 2.1; white-space: pre-wrap;
    font-size: 14px; color: #6b7280;
}

/* 첨삭 마크업 */
.del {
    background: #fee2e2; color: #991b1b;
    text-decoration: line-through;
    border-radius: 4px; padding: 1px 4px;
    font-size: 13px;
}
.ins {
    background: #dcfce7; color: #166534;
    border-radius: 4px; padding: 1px 6px;
    font-weight: 600; font-size: 13px;
}

.c-item {
    background: #f3f4f6; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 10px;
    font-size: 13px; line-height: 1.8;
}
.c-issue { color: #111827; margin-bottom: 4px; }
.c-detail {
    font-size: 12px; color: #6b7280;
    border-left: 3px solid #d1d5db;
    padding-left: 10px; margin-top: 6px;
    line-height: 1.7;
}
.c-tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700; margin-right: 8px;
}
.cmp-label {
    font-size: 12px; font-weight: 700; color: #6b7280;
    text-transform: uppercase; letter-spacing: .05em;
    margin-bottom: 8px;
}
/* 한국어 기준 */
.tag-grammar    { background: #fef3c7; color: #92400e; }
.tag-structure  { background: #dbeafe; color: #1e40af; }
.tag-vocabulary { background: #d1fae5; color: #065f46; }
.tag-logic      { background: #ede9fe; color: #5b21b6; }
.tag-theme      { background: #ffe4e6; color: #9f1239; }
/* 영어 기준 */
.tag-cohesion   { background: #ede9fe; color: #5b21b6; }
.tag-thesis     { background: #ffe4e6; color: #9f1239; }
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
.tok-info {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-radius: 10px; padding: 10px 14px;
    font-size: 12px; color: #0369a1; margin-bottom: 8px;
}
.legend {
    font-size: 12px; color: #6b7280;
    margin-bottom: 10px; line-height: 2;
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


# ── API 키 로드 ───────────────────────────────────────────────────
def _load_key(name: str) -> str:
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
    "markup_text":   "",   # ~~삭제~~(수정) 마크업 포함 원본 출력
    "edited_text":   "",   # 마크업 제거한 깔끔한 첨삭본
    "criteria":      [],
    "lang":          "ko",
    "score":         None,
    "analysis_done": False,
    "max_tokens":    4096,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── 사이드바 ──────────────────────────────────────────────────────
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
def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def recommend_tokens(text: str) -> int:
    """글자 수 기준으로 권장 토큰 반환 (마크업 오버헤드 2배 감안)."""
    char_cnt = len(text.replace(" ", "").replace("\n", ""))
    for _, limit, tokens in TOKEN_OPTIONS:
        if char_cnt <= limit:
            return tokens
    return TOKEN_OPTIONS[-1][2]


def render_markup(text: str) -> str:
    """
    Gemini 출력의 ~~삭제~~(수정) 마크업을 컬러 HTML로 변환.
    패턴:
      ~~원문~~(수정본) → 빨강 취소선 + 초록 괄호 수정
      ~~원문~~         → 빨강 취소선 (삭제)
    """
    result = ""
    # 패턴 1: ~~del~~(ins)  |  패턴 2: ~~del~~
    pattern = re.compile(r'~~(.*?)~~\(([^)]*)\)|~~(.*?)~~', re.DOTALL)
    last = 0
    for m in pattern.finditer(text):
        # 매치 전 일반 텍스트 (줄바꿈 보존)
        before = esc(text[last:m.start()]).replace("\n", "<br>")
        result += before

        if m.group(1) is not None:
            # ~~del~~(ins)
            result += (f'<span class="del">{esc(m.group(1))}</span>'
                       f'<span class="ins">({esc(m.group(2))})</span>')
        else:
            # ~~del~~ only
            result += f'<span class="del">{esc(m.group(3))}</span>'

        last = m.end()

    result += esc(text[last:]).replace("\n", "<br>")
    return result


def apply_changes(orig_text: str, changes: list) -> tuple:
    """
    Gemini의 변경 목록을 원문에 적용해 markup_text, edited_text 생성.
    changes: [{"orig": "원래 구절", "new": "수정 구절"}, ...]
    """
    # 원문에서 각 변경 위치 탐색 후 위치순 정렬
    positioned = []
    for ch in changes:
        orig = ch.get("orig", "").strip()
        new  = ch.get("new",  "").strip()
        if orig and orig in orig_text:
            positioned.append((orig_text.find(orig), len(orig), orig, new))
    positioned.sort(key=lambda x: x[0])

    markup_parts, clean_parts = [], []
    cursor = 0
    for pos, length, orig, new in positioned:
        if pos < cursor:          # 겹치는 변경은 건너뜀
            continue
        # 변경 전 그대로인 부분
        markup_parts.append(orig_text[cursor:pos])
        clean_parts.append(orig_text[cursor:pos])
        # 변경 부분
        if new:
            markup_parts.append(f"~~{orig}~~({new})")
            clean_parts.append(new)
        else:                     # 삭제
            markup_parts.append(f"~~{orig}~~")
        cursor = pos + length

    markup_parts.append(orig_text[cursor:])
    clean_parts.append(orig_text[cursor:])
    return "".join(markup_parts), "".join(clean_parts)


def img_to_b64(pil_image: Image.Image, max_px: int = 2000, quality: int = 92) -> str:
    img = pil_image.copy()
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def run_ocr(image: Image.Image) -> str:
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


def call_gemini(prompt: str, max_tokens: int = 2048, timeout: int = 90) -> str:
    """Gemini 호출. timeout 초 안에 응답 없으면 TimeoutError 발생."""
    genai.configure(api_key=GEMINI_KEY)
    gm  = genai.GenerativeModel(GEMINI_MODEL)
    cfg = genai.GenerationConfig(temperature=0.3, max_output_tokens=max_tokens)

    def _call():
        return gm.generate_content(prompt, generation_config=cfg).text

    for attempt in range(2):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise RuntimeError(
                        f"⏱️ Gemini 응답 시간 초과 ({timeout}초).\n"
                        "네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요."
                    )
        except RuntimeError:
            raise
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


def render_criteria(criteria: list, lang: str = "ko"):
    if lang == "en":
        tag_map = {
            "grammar":    ("Grammar",            "tag-grammar"),
            "structure":  ("Sentence Structure", "tag-structure"),
            "vocabulary": ("Vocabulary",         "tag-vocabulary"),
            "logic":      ("Logic & Flow",       "tag-logic"),
            "theme":      ("Theme",              "tag-theme"),
            "cohesion":   ("Cohesion",           "tag-cohesion"),
            "thesis":     ("Thesis",             "tag-thesis"),
        }
    else:
        tag_map = {
            "grammar":    ("맞춤법·문법", "tag-grammar"),
            "structure":  ("문장 구조",   "tag-structure"),
            "vocabulary": ("어휘·표현",   "tag-vocabulary"),
            "logic":      ("논리·흐름",   "tag-logic"),
            "theme":      ("주제 일관성", "tag-theme"),
            "cohesion":   ("Cohesion",    "tag-cohesion"),
            "thesis":     ("Thesis",      "tag-thesis"),
        }
    if not criteria:
        msg = "✅ No major issues found." if lang == "en" else "✅ 큰 문제가 없습니다."
        st.success(msg)
        return
    for c in criteria:
        t = c.get("type", "grammar")
        label, css = tag_map.get(t, (c.get("label", t), "tag-grammar"))
        detail_html = (f'<div class="c-detail">{esc(c["detail"])}</div>'
                       if c.get("detail") else "")
        st.markdown(
            f'<div class="c-item">'
            f'<div class="c-issue"><span class="c-tag {css}">{label}</span>{esc(c.get("issue",""))}</div>'
            f'{detail_html}'
            f'</div>', unsafe_allow_html=True,
        )


def build_txt() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    crit = "\n".join(f"• [{c.get('label','')}] {c.get('issue','')}"
                     for c in st.session_state.criteria) or "없음"
    return "\n".join([
        "글 첨삭 완성본 (Google Vision + Gemini)",
        f"작성일: {now}", "─"*40, "",
        "[원문]", st.session_state.orig_text, "", "─"*40, "",
        "[첨삭 기준]", crit, "", "─"*40, "",
        "[변경 표시 (~~삭제~~(수정) 형식)]", st.session_state.markup_text, "", "─"*40, "",
        "[첨삭 완성본]", st.session_state.edited_text, "",
    ])


def build_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    crit_html = "".join(
        f'<li><span class="tag {c.get("type","grammar")}">'
        f'{c.get("label","")}</span>{esc(c.get("issue",""))}</li>'
        for c in st.session_state.criteria
    ) or "<li>없음</li>"
    markup_html = render_markup(st.session_state.markup_text)
    css = ("body{font-family:-apple-system,sans-serif;max-width:700px;margin:40px auto;"
           "padding:20px;color:#111;line-height:1.9}"
           "h1{color:#4f46e5;border-bottom:2px solid #4f46e5;padding-bottom:8px}"
           ".date{color:#9ca3af;font-size:12px;margin-bottom:24px}"
           "h2{color:#374151;font-size:15px;margin:22px 0 10px}"
           ".box{padding:16px;border-radius:12px;font-size:14px;line-height:2}"
           ".o{background:#f9fafb;border-left:3px solid #d1d5db;color:#6b7280;white-space:pre-wrap}"
           ".e{background:#eef2ff;border-left:3px solid #4f46e5;white-space:pre-wrap}"
           ".m{background:#f9fafb;border-left:3px solid #4f46e5;line-height:2.6;word-break:break-word}"
           "ul{list-style:none;padding:0;display:flex;flex-direction:column;gap:8px}"
           "li{padding:10px 14px;background:#f3f4f6;border-radius:9px;font-size:13px}"
           ".tag{display:inline-block;padding:2px 9px;border-radius:20px;"
           "font-size:11px;font-weight:700;margin-right:7px}"
           ".grammar{background:#fef3c7;color:#92400e}"
           ".structure{background:#dbeafe;color:#1e40af}"
           ".vocabulary{background:#d1fae5;color:#065f46}"
           ".logic{background:#ede9fe;color:#5b21b6}"
           ".theme{background:#ffe4e6;color:#9f1239}"
           ".cohesion{background:#ede9fe;color:#5b21b6}"
           ".thesis{background:#ffe4e6;color:#9f1239}"
           ".del{background:#fee2e2;color:#991b1b;text-decoration:line-through;"
           "border-radius:4px;padding:1px 4px;font-size:13px}"
           ".ins{background:#dcfce7;color:#166534;border-radius:4px;"
           "padding:1px 6px;font-weight:600;font-size:13px}"
           ".legend{font-size:12px;color:#6b7280;margin-bottom:10px}")
    return (f'<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>글 첨삭 완성본</title><style>{css}</style></head><body>'
            f'<h1>✍️ 글 첨삭 완성본</h1>'
            f'<p class="date">작성일: {now} | Google Vision + Gemini</p>'
            f'<h2>📄 원문</h2><div class="box o">{esc(st.session_state.orig_text)}</div>'
            f'<h2>📋 첨삭 기준</h2><ul>{crit_html}</ul>'
            f'<h2>🔍 변경 표시</h2>'
            f'<p class="legend">'
            f'<span class="del">취소선</span> 삭제된 원문 &nbsp;'
            f'<span class="ins">(괄호)</span> 수정된 내용</p>'
            f'<div class="box m">{markup_html}</div>'
            f'<h2>✨ 첨삭 완성본</h2>'
            f'<div class="box e">{esc(st.session_state.edited_text)}</div>'
            f'</body></html>')


# ═══════════════════════════════════════════════════════════════════
# 메인 UI
# ═══════════════════════════════════════════════════════════════════

st.markdown("## ✍️ 글 첨삭 도우미 Pro")
st.markdown("Google Vision OCR · Gemini AI 첨삭")

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
                    st.session_state.ocr_text      = text
                    st.session_state.analysis_done = False
                    st.session_state.markup_text   = ""
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
    char_cnt = len(text_input.replace(" ", "").replace("\n", ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("글자수", f"{char_cnt:,}")
    c2.metric("어절수", f"{len(text_input.split()):,}")
    c3.metric("문장수", max(len(re.findall(r'[.!?。]', text_input)), 1))

    # ── 토큰 선택 ─────────────────────────────────────────────────
    st.markdown("")
    rec_tokens = recommend_tokens(text_input)
    rec_label  = next(label for label, _, tok in TOKEN_OPTIONS if tok == rec_tokens)
    tok_labels = [label for label, _, _ in TOKEN_OPTIONS]
    tok_values = {label: tok for label, _, tok in TOKEN_OPTIONS}

    sel_label = st.select_slider(
        "📊 출력 토큰 설정",
        options=tok_labels,
        value=rec_label,
        help="첨삭 마크업 포함 출력을 감안해 자동 추천합니다. 결과가 잘리면 한 단계 올려주세요.",
    )
    selected_tokens = tok_values[sel_label]
    st.session_state.max_tokens = selected_tokens

    approx_chars = selected_tokens
    st.markdown(
        f'<div class="tok-info">✅ 현재 설정: <strong>{sel_label}</strong> '
        f'— 출력 약 <strong>{approx_chars:,}자</strong> 가능 '
        f'{"&nbsp;🤖 자동 추천" if selected_tokens == rec_tokens else ""}</div>',
        unsafe_allow_html=True,
    )

if st.button(
    "🤖 AI 첨삭 시작하기 (Gemini)",
    type="primary",
    use_container_width=True,
    disabled=len(text_input.strip()) < 5,
):
    st.session_state.orig_text = text_input.strip()
    max_tok = st.session_state.max_tokens

    # API 1회 호출 — 기준 분석 + 변경사항 동시 출력
    with st.spinner("AI 첨삭 중... (최대 90초)"):
        try:
            lang_hint = ("ko:grammar(맞춤법·띄어쓰기·어미)/structure(문장구조·호응)/vocabulary(어휘·번역투·구어체)/logic(논리·흐름·접속어)/theme(주제일관성)—경어체일관성중점"
                         "|en:grammar(SVA·tense·articles)/structure(run-on·fragments)/vocabulary(repetition·register)/cohesion(transitions·flow)/thesis(clarity·support)")
            raw = call_gemini(
                f"""글쓰기 첨삭 전문가. 원문 분석 후 JSON만 출력.

원문:"{st.session_state.orig_text}"

언어자동감지→기준적용: {lang_hint}

출력(JSON만,코드블록없이):
{{"lang":"ko|en","score":0-100,"criteria":[{{"type":"grammar|structure|vocabulary|logic|theme|cohesion|thesis","label":"라벨","issue":"문제요약","detail":"한국어1문장40자이내,수정전→후예시"}}],"changes":[{{"orig":"원문정확구절","new":"수정"}}]}}

규칙:criteria실제문제최대5개,changes전체빠짐없이,orig=원문정확복사,JSON만""",
                max_tokens=max_tok,
            )

            cleaned = re.sub(r"```json?|```", "", raw).strip()
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if not m:
                st.error("JSON 파싱 실패 — Gemini 원문 응답:")
                st.code(raw[:500] if raw else "(빈 응답)", language="text")
                st.caption("💡 토큰 슬라이더를 한 단계 올려서 다시 시도해주세요.")
                st.stop()
            data = json.loads(m.group())

            st.session_state.score    = data.get("score")
            st.session_state.lang     = data.get("lang", "ko")
            st.session_state.criteria = data.get("criteria", [])
            changes = data.get("changes", [])

            markup, clean = apply_changes(st.session_state.orig_text, changes)
            st.session_state.markup_text   = markup
            st.session_state.edited_text   = clean
            st.session_state.analysis_done = True

        except json.JSONDecodeError as e:
            st.error(f"JSON 형식 오류: {e}")
            st.code(raw[:500], language="text")
            st.caption("💡 토큰 슬라이더를 한 단계 올려서 다시 시도해주세요.")
            st.stop()
        except Exception as e:
            st.error(f"첨삭 오류: {e}")
            st.caption("💡 API 키가 올바른지, 네트워크가 연결되어 있는지 확인해주세요.")
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

    lang = st.session_state.get("lang", "ko")
    lang_badge = "🇰🇷 한국어" if lang == "ko" else "🇺🇸 English"
    with st.expander(f"📋 첨삭 기준 ({lang_badge})", expanded=True):
        render_criteria(st.session_state.criteria, lang)

    tab_cmp, tab_final = st.tabs(["📄 원문 vs 첨삭본", "✨ 최종 완성본"])

    with tab_cmp:
        # 원문(좌) | 첨삭본(우) 나란히 비교
        st.markdown(
            '<div class="legend">'
            '<span class="del">빨강 취소선</span> 삭제된 원문 &nbsp;&nbsp;'
            '<span class="ins">(초록 괄호)</span> 수정된 내용</div>',
            unsafe_allow_html=True,
        )
        col_o, col_e = st.columns(2)
        with col_o:
            st.markdown('<p class="cmp-label">원문</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="orig-box" style="min-height:200px">'
                f'{esc(st.session_state.orig_text)}</div>',
                unsafe_allow_html=True,
            )
        with col_e:
            st.markdown('<p class="cmp-label">첨삭본</p>', unsafe_allow_html=True)
            markup_html = render_markup(st.session_state.markup_text)
            st.markdown(
                f'<div class="markup-box" style="min-height:200px">{markup_html}</div>',
                unsafe_allow_html=True,
            )

    with tab_final:
        # 수정사항만 반영된 깔끔한 최종 완성본
        st.markdown(
            f'<div class="result-box">{esc(st.session_state.edited_text)}</div>',
            unsafe_allow_html=True,
        )

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
