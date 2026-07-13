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
from datetime import datetime

import streamlit as st
import requests
from PIL import Image
import google.generativeai as genai

# ── 고정 상수 ─────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"

# 토큰 옵션 — 마크업 오버헤드(~2배)를 감안하여 글자수 기준을 보수적으로 설정
TOKEN_OPTIONS = [
    ("소형 · 2,048 토큰",  300,   2048),
    ("중형 · 4,096 토큰",  800,   4096),
    ("대형 · 8,192 토큰",  2000,  8192),
    ("특대 · 16,384 토큰", 99999, 16384),
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


def strip_markup(text: str) -> str:
    """마크업 제거 → 깔끔한 첨삭본 생성."""
    # ~~del~~(ins) → ins
    text = re.sub(r'~~.*?~~\(([^)]*)\)', r'\1', text, flags=re.DOTALL)
    # ~~del~~ → (삭제)
    text = re.sub(r'~~.*?~~', '', text, flags=re.DOTALL)
    return text.strip()


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


def call_gemini(prompt: str, max_tokens: int = 2048) -> str:
    genai.configure(api_key=GEMINI_KEY)
    gm  = genai.GenerativeModel(GEMINI_MODEL)
    cfg = genai.GenerationConfig(temperature=0.3, max_output_tokens=max_tokens)

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
    tag_cls = {"grammar":"grammar","style":"style","logic":"logic","flow":"flow"}
    crit_html = "".join(
        f'<li><span class="tag {tag_cls.get(c.get("type","style"),"style")}">'
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
           ".grammar{background:#fef3c7;color:#92400e}.style{background:#dbeafe;color:#1e40af}"
           ".logic{background:#d1fae5;color:#065f46}.flow{background:#ede9fe;color:#5b21b6}"
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

    # 1단계 — 첨삭 기준 분석
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
- 실제 문제만 포함 (없으면 [])  - 최대 5개  - JSON만 출력""",
                max_tokens=1024,
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

    # 2단계 — 전체 첨삭 (마크업 포함)
    with st.spinner(f"(2/2) 전체 첨삭 중... (출력 토큰: {max_tok:,})"):
        try:
            crit_str = ("\n".join(
                f"- [{c.get('label','')}] {c.get('issue','')}"
                for c in st.session_state.criteria
            ) or "- 전반적으로 자연스럽고 완성도 높게 개선")

            edit_raw = call_gemini(
                f"""당신은 전문 글쓰기 첨삭 선생님입니다.
아래 글을 처음 문장부터 마지막 문장까지 단 한 줄도 빠짐없이 전부 첨삭하세요.

【원문】
\"\"\"
{st.session_state.orig_text}
\"\"\"

【첨삭 기준】
{crit_str}

【출력 규칙 — 반드시 준수】
1. 원문 전체를 순서대로 처음부터 끝까지 모두 출력하세요.
2. 수정한 부분은 아래 형식으로 표기하세요:
   - 교체: ~~원래 표현~~(수정된 표현)
   - 삭제: ~~삭제할 내용~~
   - 추가: 새 내용을 그냥 삽입 (표기 없음)
3. 수정이 없는 부분은 그대로 출력하세요.
4. 모든 문단을 빠짐없이 처리하고, 도중에 멈추지 마세요.
5. 마지막 줄에 ===점수===숫자 (예: ===점수===72) 를 추가하세요.
6. 마크업과 글 내용 외에 설명·제목·머리말은 일절 출력하지 마세요.""",
                max_tokens=max_tok,
            )

            score_m = re.search(r"===점수===(\d+)", edit_raw)
            raw_clean = re.sub(r"===점수===\d+", "", edit_raw).strip()

            st.session_state.score       = int(score_m.group(1)) if score_m else None
            st.session_state.markup_text = raw_clean
            st.session_state.edited_text = strip_markup(raw_clean)
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

    tab_markup, tab_edit, tab_cmp = st.tabs(["🔍 변경 표시", "✨ 첨삭본", "📄 원문 비교"])

    with tab_markup:
        st.markdown(
            '<div class="legend">'
            '<span class="del">빨강 취소선</span> 삭제된 원문 &nbsp;&nbsp;'
            '<span class="ins">(초록 괄호)</span> 수정된 내용</div>',
            unsafe_allow_html=True,
        )
        markup_html = render_markup(st.session_state.markup_text)
        st.markdown(f'<div class="markup-box">{markup_html}</div>', unsafe_allow_html=True)

    with tab_edit:
        st.markdown(
            f'<div class="result-box">{esc(st.session_state.edited_text)}</div>',
            unsafe_allow_html=True,
        )

    with tab_cmp:
        co, ce = st.columns(2)
        with co:
            st.markdown("**원문**")
            st.markdown(
                f'<div class="orig-box">{esc(st.session_state.orig_text)}</div>',
                unsafe_allow_html=True,
            )
        with ce:
            st.markdown("**첨삭본**")
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
