"""
글 첨삭 도우미 Pro — Streamlit 버전
Google Cloud Vision API (OCR) + Google Gemini API (AI 첨삭)

실행: streamlit run app.py
배포: https://share.streamlit.io
"""

import streamlit as st
import requests
import base64
import io
import json
import re
from PIL import Image
from datetime import datetime

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
/* 컨텐츠 최대 너비 */
.block-container { max-width: 720px; padding-top: 1.5rem; padding-bottom: 3rem; }

/* 결과 박스 */
.result-box {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    border-radius: 12px;
    padding: 18px 20px;
    line-height: 2.1;
    white-space: pre-wrap;
    font-size: 14px;
    color: #111827;
    margin-bottom: 4px;
}
.orig-box {
    background: #f9fafb;
    border-left: 4px solid #d1d5db;
    border-radius: 12px;
    padding: 18px 20px;
    line-height: 2.1;
    white-space: pre-wrap;
    font-size: 14px;
    color: #6b7280;
}

/* 첨삭 기준 아이템 */
.c-item {
    background: #f3f4f6;
    border-radius: 10px;
    padding: 11px 15px;
    margin-bottom: 8px;
    font-size: 13px;
    line-height: 1.7;
}
.c-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 8px;
}
.tag-grammar { background: #fef3c7; color: #92400e; }
.tag-style   { background: #dbeafe; color: #1e40af; }
.tag-logic   { background: #d1fae5; color: #065f46; }
.tag-flow    { background: #ede9fe; color: #5b21b6; }

/* 점수 뱃지 */
.score-pill {
    display: inline-block;
    padding: 5px 20px;
    border-radius: 20px;
    font-weight: 800;
    font-size: 15px;
    color: white;
}

/* 스텝 번호 */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #4f46e5;
    color: white;
    font-size: 13px;
    font-weight: 800;
    margin-right: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


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


# ── 사이드바: API 키 입력 ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API 설정")
    st.caption("API 키는 서버에서만 사용되며 저장되지 않습니다.")

    # secrets.toml 에 미리 설정된 경우 자동 로드
    try:
        vision_default = st.secrets["VISION_API_KEY"]
    except Exception:
        vision_default = ""
    try:
        gemini_default = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_default = ""

    vision_key = st.text_input(
        "Google Vision API Key",
        value=vision_default,
        type="password",
        placeholder="AIzaSy...",
    )
    gemini_key = st.text_input(
        "Gemini API Key",
        value=gemini_default,
        type="password",
        placeholder="AIzaSy...",
    )

    gemini_model = st.selectbox(
        "Gemini 모델",
        options=[
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ],
        index=0,
        help="무료 티어 추천: gemini-1.5-flash (일 1,500회 무료)",
    )
    st.caption({
        "gemini-1.5-flash":    "✅ 무료 1,500회/일 — 추천",
        "gemini-1.5-flash-8b": "✅ 무료 가장 넉넉",
        "gemini-2.0-flash":    "⚠️ 무료 한도 매우 적음",
        "gemini-1.5-pro":      "💎 고품질, 무료 50회/일",
    }.get(gemini_model, ""))

    keys_ok = bool(vision_key) and bool(gemini_key)

    if keys_ok:
        st.success("✅ API 연결됨")
    else:
        st.info("위에 API 키를 입력하세요")

    with st.expander("📌 API 키 발급 방법"):
        st.markdown("""
**Google Vision API** (OCR)
1. [console.cloud.google.com](https://console.cloud.google.com) 접속
2. 프로젝트 생성 → APIs & Services → Library
3. **"Cloud Vision API"** 검색 → 사용 설정
4. Credentials → **Create Credentials → API Key**

**Gemini API** (AI 첨삭)
1. [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) 접속
2. **"Create API key"** 클릭 → 복사

둘 다 무료 티어로 충분합니다.
        """)

    with st.expander("☁️ Streamlit Cloud 배포 팁"):
        st.markdown("""
**API 키를 코드에 넣지 않고 배포하려면:**

1. [share.streamlit.io](https://share.streamlit.io) 에서 앱 배포
2. 앱 대시보드 → **Settings → Secrets** 탭
3. 아래 내용 붙여넣기:
```toml
VISION_API_KEY = "여기에_Vision_키"
GEMINI_API_KEY = "여기에_Gemini_키"
```
→ 사이드바 입력 없이 자동 로드됩니다.
        """)

    st.divider()

    if st.button("🔄 처음부터 다시", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.caption("✍️ 글 첨삭 도우미 Pro\nGoogle Vision + Gemini AI")


# ── 유틸 함수 ────────────────────────────────────────────────────
def img_to_b64(pil_image: Image.Image, max_px: int = 2000, quality: int = 92) -> str:
    """PIL 이미지 → JPEG base64 (크기 압축 포함)"""
    img = pil_image.copy()
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def run_ocr(image: Image.Image, api_key: str) -> str:
    """Google Cloud Vision API — DOCUMENT_TEXT_DETECTION"""
    b64 = img_to_b64(image)
    resp = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
        json={
            "requests": [{
                "image": {"content": b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["ko", "en"]},
            }]
        },
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


def call_gemini(prompt: str, api_key: str, model: str = "gemini-1.5-flash") -> str:
    """Gemini API 호출 (모델 선택 가능, 할당량 오류 친절 처리)"""
    # 1.5 계열은 v1, 2.0 계열은 v1beta 엔드포인트 사용
    api_version = "v1beta" if "2.0" in model else "v1"
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/{api_version}/models/"
        f"{model}:generateContent?key={api_key}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
        },
        timeout=60,
    )
    data = resp.json()
    if "error" in data:
        msg = data["error"].get("message", "")
        # 할당량 초과 — 사용자 친화적 메시지
        if "quota" in msg.lower() or "429" in str(data["error"].get("code", "")):
            import re as _re
            retry_sec = _re.search(r"retry in ([\d.]+)s", msg)
            wait_msg = f" ({retry_sec.group(1)}초 후 재시도)" if retry_sec else ""
            raise RuntimeError(
                f"Gemini 무료 할당량 초과{wait_msg}.\n"
                f"왼쪽 사이드바에서 **gemini-1.5-flash** 또는 **gemini-1.5-flash-8b** 로 변경하거나, "
                f"잠시 후 다시 시도해주세요."
            )
        raise RuntimeError(msg)
    return data["candidates"][0]["content"]["parts"][0]["text"]


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
        t = c.get("type", "style")
        label, css = tag_map.get(t, (c.get("label", ""), "tag-style"))
        issue = c.get("issue", "")
        st.markdown(
            f'<div class="c-item">'
            f'<span class="c-tag {css}">{label}</span>{issue}'
            f'</div>',
            unsafe_allow_html=True,
        )


def build_txt() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    crit_lines = (
        "\n".join(f"• [{c.get('label','')}] {c.get('issue','')}"
                  for c in st.session_state.criteria)
        or "없음"
    )
    return "\n".join([
        "글 첨삭 완성본 (Google Vision + Gemini)",
        f"작성일: {now}",
        "─" * 40, "",
        "[원문]", st.session_state.orig_text, "",
        "─" * 40, "",
        "[첨삭 기준]", crit_lines, "",
        "─" * 40, "",
        "[첨삭 완성본]", st.session_state.edited_text, "",
    ])


def build_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    tag_map = {"grammar": "grammar", "style": "style", "logic": "logic", "flow": "flow"}

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    crit_html = "".join(
        f'<li><span class="tag {tag_map.get(c.get("type","style"),"style")}">'
        f'{c.get("label","")}</span>{esc(c.get("issue",""))}</li>'
        for c in st.session_state.criteria
    ) or "<li>없음</li>"

    css = (
        "body{font-family:-apple-system,sans-serif;max-width:700px;margin:40px auto;"
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
        ".logic{background:#d1fae5;color:#065f46}.flow{background:#ede9fe;color:#5b21b6}"
    )
    return (
        f'<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>글 첨삭 완성본</title><style>{css}</style></head><body>'
        f'<h1>✍️ 글 첨삭 완성본</h1>'
        f'<p class="date">작성일: {now} | Google Vision + Gemini AI</p>'
        f'<h2>📄 원문</h2><div class="box o">{esc(st.session_state.orig_text)}</div>'
        f'<h2>📋 첨삭 기준</h2><ul>{crit_html}</ul>'
        f'<h2>✨ 첨삭 완성본</h2><div class="box e">{esc(st.session_state.edited_text)}</div>'
        f'</body></html>'
    )


# ═══════════════════════════════════════════════════════════════════
# 메인 UI
# ═══════════════════════════════════════════════════════════════════

st.markdown("## ✍️ 글 첨삭 도우미 Pro")
st.markdown("Google Vision OCR · Gemini AI 첨삭 · 서버 사이드 처리 (방화벽 무관)")

st.divider()

# ──────────────────────────────────────────────────────────────────
# STEP 1 — 사진 업로드 & OCR
# ──────────────────────────────────────────────────────────────────
st.markdown('<span class="step-num">1</span> **사진 업로드 & 텍스트 인식**', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "사진 파일을 업로드하세요",
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

        if not keys_ok:
            st.warning("왼쪽 사이드바에 API 키를 먼저 입력하세요.")
        else:
            if st.button("🔍 텍스트 인식 시작", type="primary", use_container_width=True):
                with st.spinner("Google Vision OCR 실행 중..."):
                    try:
                        text = run_ocr(image, vision_key)
                        st.session_state.ocr_text = text
                        # 이전 분석 결과 초기화
                        st.session_state.analysis_done = False
                        st.session_state.edited_text = ""
                        st.session_state.criteria = []
                        st.session_state.score = None
                        st.toast(f"✅ {len(text.replace(' ','').replace(chr(10),''))}자 인식 완료!")
                        st.rerun()
                    except Exception as e:
                        msg = str(e)
                        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
                            st.error("Vision API 키가 유효하지 않습니다. 키를 다시 확인해주세요.")
                        else:
                            st.error(f"OCR 오류: {msg}")

st.divider()

# ──────────────────────────────────────────────────────────────────
# STEP 2 — 텍스트 확인 & AI 첨삭
# ──────────────────────────────────────────────────────────────────
st.markdown('<span class="step-num">2</span> **텍스트 확인 & AI 첨삭**', unsafe_allow_html=True)

if st.session_state.ocr_text:
    st.success(f"✅ OCR 완료 — {len(st.session_state.ocr_text.replace(' ','').replace(chr(10),''))}자 인식됨")

text_input = st.text_area(
    "첨삭할 글",
    value=st.session_state.ocr_text,
    height=260,
    placeholder=(
        "여기에 텍스트를 직접 입력하거나,\n"
        "위에서 사진을 업로드하면 자동으로 채워집니다."
    ),
    label_visibility="collapsed",
)

if text_input.strip():
    chars = len(text_input.replace(" ", "").replace("\n", ""))
    words = len(text_input.split())
    sents = max(len(re.findall(r"[.!?。]", text_input)), 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("글자수", f"{chars:,}")
    c2.metric("어절수", f"{words:,}")
    c3.metric("문장수", sents)

can_analyze = keys_ok and len(text_input.strip()) >= 5

if st.button(
    "🤖 AI 첨삭 시작하기 (Gemini)",
    type="primary",
    use_container_width=True,
    disabled=not can_analyze,
):
    st.session_state.orig_text = text_input.strip()

    # ── 1/2: 첨삭 기준 분석 ──────────────────────────────────────
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

type 값: grammar(문법·맞춤법), style(문체·표현), logic(논리·내용), flow(흐름·구성)
- 실제로 있는 문제만 포함 (없으면 빈 배열 [])
- 최대 5개
- JSON만 출력, 다른 텍스트 없음""",
                gemini_key,
                gemini_model,
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

    # ── 2/2: 첨삭 완성본 작성 ────────────────────────────────────
    with st.spinner("(2/2) 첨삭 완성본 작성 중..."):
        try:
            crit_str = (
                "\n".join(
                    f"- [{c.get('label','')}] {c.get('issue','')}"
                    for c in st.session_state.criteria
                )
                or "- 전반적으로 자연스럽고 완성도 높게 개선"
            )
            edit_raw = call_gemini(
                f"""전문 글쓰기 첨삭 선생님으로서 아래 글을 첨삭해주세요.

원문:
\"\"\"
{st.session_state.orig_text}
\"\"\"

첨삭 기준:
{crit_str}

규칙:
1. 원문의 핵심 내용과 의도를 반드시 유지
2. 문법·맞춤법 오류 수정
3. 자연스러운 문체로 개선
4. 논리적 흐름 강화
5. 마지막 줄에 ===점수===숫자 형태로 원문 완성도 점수 표기 (예: ===점수===72)
6. 첨삭된 글만 출력 (설명 없음, 마크다운 없음)""",
                gemini_key,
                gemini_model,
            )
            score_m = re.search(r"===점수===(\d+)", edit_raw)
            st.session_state.score = int(score_m.group(1)) if score_m else None
            st.session_state.edited_text = re.sub(r"===점수===\d+", "", edit_raw).strip()
            st.session_state.analysis_done = True
        except Exception as e:
            st.error(f"첨삭 오류: {e}")
            st.stop()

    st.rerun()

if not can_analyze and not keys_ok:
    st.caption("👈 사이드바에 API 키를 입력한 뒤 시작하세요.")

st.divider()

# ──────────────────────────────────────────────────────────────────
# STEP 3 — 결과 & 저장
# ──────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    st.markdown('<span class="step-num">3</span> **첨삭 결과**', unsafe_allow_html=True)

    # 점수 표시
    if st.session_state.score is not None:
        s = st.session_state.score
        emoji  = "🏆" if s >= 90 else "⭐" if s >= 75 else "👍" if s >= 60 else "📝"
        color  = "#10b981" if s >= 90 else "#4f46e5" if s >= 75 else "#f59e0b" if s >= 60 else "#6b7280"
        st.markdown(
            f'원문 완성도: '
            f'<span class="score-pill" style="background:{color}">{emoji} {s}점</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    # 첨삭 기준
    with st.expander("📋 첨삭 기준", expanded=True):
        render_criteria(st.session_state.criteria)

    # 결과 탭
    tab_edit, tab_compare = st.tabs(["✨ 첨삭본", "📄 원문 비교"])

    with tab_edit:
        st.markdown(
            f'<div class="result-box">{st.session_state.edited_text}</div>',
            unsafe_allow_html=True,
        )

    with tab_compare:
        col_o, col_e = st.columns(2)
        with col_o:
            st.markdown("**원문**")
            st.markdown(
                f'<div class="orig-box">{st.session_state.orig_text}</div>',
                unsafe_allow_html=True,
            )
        with col_e:
            st.markdown("**첨삭본**")
            st.markdown(
                f'<div class="result-box">{st.session_state.edited_text}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # 저장
    st.markdown("**💾 저장**")
    fname = st.text_input("파일 이름", value="첨삭완성본", label_visibility="collapsed")

    col_txt, col_html = st.columns(2)
    with col_txt:
        st.download_button(
            "📄 TXT 저장",
            data=build_txt().encode("utf-8"),
            file_name=f"{fname}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_html:
        st.download_button(
            "🌐 HTML 저장",
            data=build_html().encode("utf-8"),
            file_name=f"{fname}.html",
            mime="text/html",
            use_container_width=True,
        )
