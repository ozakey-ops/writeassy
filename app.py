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
import concurrent.futures
from datetime import datetime

import streamlit as st
import requests
from PIL import Image
import google.generativeai as genai

# ── 고정 상수 ─────────────────────────────────────────────────────
MODEL_FAST     = "gemini-3.1-flash-lite"  # 문법 교정 (빠름)
MODEL_FULL     = "gemini-3.5-flash"       # 윤문 첨삭 (정확)

# 토큰 자동 증가 단계: 1024 → 2048 → 4096 → 8192 → 확인 후 진행
TOKEN_LEVELS   = [1024, 2048, 4096, 8192]
EXTENDED_TOKENS = 16384

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
/* 공통 기준 태그 색상 (첨삭 마크업 색상과 동일) */
.tag-grammar       { background: #fef3c7; color: #92400e; }
.tag-structure     { background: #dbeafe; color: #1e40af; }
.tag-vocabulary    { background: #d1fae5; color: #065f46; }
.tag-logic         { background: #ede9fe; color: #5b21b6; }
.tag-theme         { background: #ffe4e6; color: #9f1239; }
/* 한국어 신규 10기준 */
.tag-comprehension { background: #fdf4ff; color: #7e22ce; }
.tag-conditions    { background: #fff7ed; color: #c2410c; }
.tag-critical      { background: #f0fdfa; color: #0f766e; }
.tag-unity         { background: #eff6ff; color: #1d4ed8; }
/* 영어 기준 (10가지) */
.tag-cohesion      { background: #ede9fe; color: #5b21b6; }
.tag-thesis        { background: #ffe4e6; color: #9f1239; }
.tag-spelling      { background: #fef9c3; color: #713f12; }
.tag-clarity       { background: #cffafe; color: #0e7490; }
.tag-organization  { background: #dcfce7; color: #166534; }
.tag-repetition    { background: #fce7f3; color: #9d174d; }
.tag-conciseness   { background: #f5f3ff; color: #6d28d9; }
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

# Gemini SDK 초기화 (앱 시작 시 1회)
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)


# ── 세션 상태 초기화 ──────────────────────────────────────────────
_defaults = {
    "ocr_text":               "",
    "orig_text":              "",
    "markup_text":            "",   # ~~삭제~~(수정) 마크업 (TXT 저장용)
    "markup_html":            "",   # 기준 유형별 컬러 HTML 마크업 (화면 표시용)
    "edited_text":            "",   # 마크업 제거한 깔끔한 첨삭본
    "criteria":               [],
    "lang":                   "ko",
    "score":                  None,
    "analysis_done":          False,
    "correction_level":       "full",   # "fast" | "full"
    "needs_extended_confirm": False,    # 8192 초과 시 사용자 확인 대기
    "used_tokens":            None,     # 실제 사용된 토큰 수
    "summary":                "",      # 첨삭 내용 전체 해설
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
        _cur_model = MODEL_FAST if st.session_state.get("correction_level") == "fast" else MODEL_FULL
        st.caption(f"모델: `{_cur_model}`")
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

def _cc(text: str) -> int:
    """공백·줄바꿈 제외 글자 수 (char count)."""
    return len(text.replace(" ", "").replace("\n", ""))


def detect_lang_ratio(text: str) -> float:
    """한국어(CJK) 비율 반환 (0.0=영어, 1.0=완전한국어)."""
    if not text:
        return 1.0
    ko = sum(1 for c in text if '가' <= c <= '힣' or 'ㄱ' <= c <= 'ㆎ')
    return ko / max(_cc(text), 1)


def estimate_start_tokens(char_count: int, level: str, text: str = "") -> int:
    """
    글자 수, 첨삭 수준, 언어 비율 기반으로 시작 토큰 계산.
    - 한국어: 글자당 ~2 토큰  (SentencePiece 기준)
    - 영어:   글자당 ~0.25 토큰 (4자당 1 토큰)
    - fast(lite) 기준 추정 후, full(flash)은 4배 적용
      이유: 영어 10기준(최대 5개) + reason + summary → 출력량 ~4배
    """
    ko_ratio     = detect_lang_ratio(text) if text else 1.0
    tok_per_char = ko_ratio * 2.0 + (1 - ko_ratio) * 0.25

    # fast 기준 추정 (criteria≤3, summary 1문장, 오버헤드 350)
    fast_est = int(char_count * tok_per_char * 0.4) + 350

    # full은 4배 (criteria≤5 + reason + 2문장 summary + 영어 10기준 레이블)
    estimated = fast_est if level == "fast" else fast_est * 4

    for level_tok in TOKEN_LEVELS:
        if level_tok >= estimated:
            return level_tok
    return TOKEN_LEVELS[-1]


# 첨삭 기준 유형별 색상 (bg, fg) — 태그 색상과 동일하게 유지
TYPE_COLORS: dict[str, tuple[str, str]] = {
    "grammar":       ("#fef3c7", "#92400e"),
    "structure":     ("#dbeafe", "#1e40af"),
    "vocabulary":    ("#d1fae5", "#065f46"),
    "logic":         ("#ede9fe", "#5b21b6"),
    "theme":         ("#ffe4e6", "#9f1239"),
    "comprehension": ("#fdf4ff", "#7e22ce"),
    "conditions":    ("#fff7ed", "#c2410c"),
    "critical":      ("#f0fdfa", "#0f766e"),
    "unity":         ("#eff6ff", "#1d4ed8"),
    "cohesion":      ("#ede9fe", "#5b21b6"),
    "thesis":        ("#ffe4e6", "#9f1239"),
    "spelling":      ("#fef9c3", "#713f12"),
    "clarity":       ("#cffafe", "#0e7490"),
    "organization":  ("#dcfce7", "#166534"),
    "repetition":    ("#fce7f3", "#9d174d"),
    "conciseness":   ("#f5f3ff", "#6d28d9"),
}
_DEL_DEFAULT = ("#fee2e2", "#991b1b")


def render_markup(text: str) -> str:
    """
    ~~삭제~~(수정) 텍스트 마크업 → HTML (TXT 저장본 열람 시 사용).
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
    Gemini 변경 목록을 원문에 적용.
    changes: [{"orig": "...", "new": "...", "type": "grammar"}, ...]
    Returns: (markup_text, markup_html, edited_text)
      - markup_text : ~~삭제~~(수정) 텍스트 형식 (TXT 내보내기용)
      - markup_html : 기준 유형별 컬러 인라인 스타일 HTML (화면 표시용)
      - edited_text : 마크업 없는 최종 첨삭본
    """
    positioned = []
    for ch in changes:
        orig  = ch.get("orig", "").strip()
        new   = ch.get("new",  "").strip()
        ctype = ch.get("type", "")
        if orig and orig in orig_text:
            positioned.append((orig_text.find(orig), len(orig), orig, new, ctype))
    positioned.sort(key=lambda x: x[0])

    markup_parts, html_parts, clean_parts = [], [], []
    cursor = 0
    for pos, length, orig, new, ctype in positioned:
        if pos < cursor:
            continue
        unchanged = orig_text[cursor:pos]
        markup_parts.append(unchanged)
        html_parts.append(esc(unchanged).replace("\n", "<br>"))
        clean_parts.append(unchanged)

        bg, fg = TYPE_COLORS.get(ctype, _DEL_DEFAULT)
        del_style = (f"background:{bg};color:{fg};text-decoration:line-through;"
                     f"border-radius:4px;padding:1px 4px;font-size:13px")
        ins_style = (f"background:{bg};color:{fg};"
                     f"border-radius:4px;padding:1px 6px;font-weight:700;font-size:13px")
        if new:
            markup_parts.append(f"~~{orig}~~({new})")
            html_parts.append(
                f'<span style="{del_style}">{esc(orig)}</span>'
                f'<span style="{ins_style}">&nbsp;{esc(new)}&nbsp;</span>'
            )
            clean_parts.append(new)
        else:
            markup_parts.append(f"~~{orig}~~")
            html_parts.append(f'<span style="{del_style}">{esc(orig)}</span>')
        cursor = pos + length

    tail = orig_text[cursor:]
    markup_parts.append(tail)
    html_parts.append(esc(tail).replace("\n", "<br>"))
    clean_parts.append(tail)

    return "".join(markup_parts), "".join(html_parts), "".join(clean_parts)


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


def call_gemini(prompt: str, max_tokens: int = 2048, timeout: int = 90,
                model: str = MODEL_FULL) -> str:
    """Gemini 호출. timeout 초 안에 응답 없으면 TimeoutError 발생."""
    gm  = genai.GenerativeModel(model)
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
            # 기존
            "grammar":      ("① Grammar",          "tag-grammar"),
            "structure":    ("④ Structure",         "tag-structure"),
            "vocabulary":   ("⑤ Vocabulary",        "tag-vocabulary"),
            "logic":        ("⑥ Logic & Flow",      "tag-logic"),
            "theme":        ("⑦ Topic",             "tag-theme"),
            "cohesion":     ("⑥ Cohesion",          "tag-cohesion"),
            "thesis":       ("⑦ Thesis",            "tag-thesis"),
            # 신규 10기준
            "spelling":     ("② Spelling",          "tag-spelling"),
            "clarity":      ("③ Clarity",           "tag-clarity"),
            "organization": ("⑧ Organization",      "tag-organization"),
            "repetition":   ("⑨ Repetition",        "tag-repetition"),
            "conciseness":  ("⑩ Conciseness",       "tag-conciseness"),
        }
    else:
        tag_map = {
            # 한국어 10기준 (full 모드)
            "theme":         ("① 주제 명확성",  "tag-theme"),
            "comprehension": ("② 제시문 이해",  "tag-comprehension"),
            "logic":         ("③ 논리 전개",    "tag-logic"),
            "organization":  ("④ 문단 구성",    "tag-organization"),
            "clarity":       ("⑤ 문장 표현",    "tag-clarity"),
            "vocabulary":    ("⑥ 어휘 선택",    "tag-vocabulary"),
            "grammar":       ("⑦ 맞춤법·문법",  "tag-grammar"),
            "conditions":    ("⑧ 시험 조건",    "tag-conditions"),
            "critical":      ("⑨ 비판적 사고",  "tag-critical"),
            "unity":         ("⑩ 완성도·통일",  "tag-unity"),
            # 구버전 / 영어 혼용 fallback
            "structure":    ("문장 구조",       "tag-structure"),
            "cohesion":     ("Cohesion",        "tag-cohesion"),
            "thesis":       ("Thesis",          "tag-thesis"),
            "spelling":     ("철자·구두점",     "tag-spelling"),
            "repetition":   ("반복 표현",       "tag-repetition"),
            "conciseness":  ("간결성",          "tag-conciseness"),
        }
    if not criteria:
        msg = "✅ No major issues found." if lang == "en" else "✅ 큰 문제가 없습니다."
        st.success(msg)
        return
    for c in criteria:
        t = c.get("type", "grammar")
        label, css = tag_map.get(t, (c.get("label", t), "tag-grammar"))
        # reason: 교정 이유 해설 (신규) / detail: 하위 호환 (구버전 출력 대비)
        reason_text = c.get("reason") or c.get("detail", "")
        reason_html = (
            f'<div class="c-detail">'
            f'<span style="font-size:10px;color:#9ca3af;margin-right:4px">💡 이유</span>'
            f'{esc(reason_text)}</div>'
        ) if reason_text else ""
        st.markdown(
            f'<div class="c-item">'
            f'<div class="c-issue"><span class="c-tag {css}">{label}</span>{esc(c.get("issue",""))}</div>'
            f'{reason_html}'
            f'</div>', unsafe_allow_html=True,
        )


def build_txt() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    def _crit_line(c):
        base = f"• [{c.get('label','')}] {c.get('issue','')}"
        reason = c.get("reason") or c.get("detail", "")
        return base + (f"\n  └ 이유: {reason}" if reason else "")
    crit = "\n".join(_crit_line(c) for c in st.session_state.criteria) or "없음"
    level_lbl = "문법 교정" if st.session_state.correction_level == "fast" else "윤문 첨삭"
    parts = [
        "글 첨삭 완성본 (Google Vision + Gemini)",
        f"작성일: {now}  |  첨삭 수준: {level_lbl}", "─"*40, "",
        "[원문]", st.session_state.orig_text, "", "─"*40, "",
    ]
    if st.session_state.get("summary"):
        parts += ["[첨삭 해설]", st.session_state.summary, "", "─"*40, ""]
    parts += [
        "[첨삭 기준]", crit, "", "─"*40, "",
        "[변경 표시 (~~삭제~~(수정) 형식)]", st.session_state.markup_text, "", "─"*40, "",
        "[첨삭 완성본]", st.session_state.edited_text, "",
    ]
    return "\n".join(parts)


def build_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    crit_html = "".join(
        f'<li><span class="tag {c.get("type","grammar")}">'
        f'{c.get("label","")}</span>{esc(c.get("issue",""))}</li>'
        for c in st.session_state.criteria
    ) or "<li>없음</li>"
    markup_html = st.session_state.get("markup_html") or render_markup(st.session_state.markup_text)
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
           ".comprehension{background:#fdf4ff;color:#7e22ce}"
           ".conditions{background:#fff7ed;color:#c2410c}"
           ".critical{background:#f0fdfa;color:#0f766e}"
           ".unity{background:#eff6ff;color:#1d4ed8}"
           ".spelling{background:#fef9c3;color:#713f12}"
           ".clarity{background:#cffafe;color:#0e7490}"
           ".organization{background:#dcfce7;color:#166534}"
           ".repetition{background:#fce7f3;color:#9d174d}"
           ".conciseness{background:#f5f3ff;color:#6d28d9}"
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
                    # 이전 분석 결과 전체 초기화
                    for _k, _v in _defaults.items():
                        if _k != "correction_level":   # 선택한 수준은 유지
                            st.session_state[_k] = _v
                    st.session_state.ocr_text = text
                    st.toast(f"✅ {_cc(text)}자 인식 완료!")
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
    st.success(f"✅ OCR 완료 — {_cc(st.session_state.ocr_text)}자 인식됨")

text_input = st.text_area(
    "첨삭할 글",
    value=st.session_state.ocr_text,
    height=260,
    placeholder="여기에 텍스트를 직접 입력하거나,\n위에서 사진을 업로드하면 자동으로 채워집니다.",
    label_visibility="collapsed",
)

_char_cnt = _cc(text_input) if text_input.strip() else 0
if _char_cnt:
    c1, c2, c3 = st.columns(3)
    c1.metric("글자수", f"{_char_cnt:,}")
    c2.metric("어절수", f"{len(text_input.split()):,}")
    c3.metric("문장수", max(len(re.findall(r'[.!?。]', text_input)), 1))

# ── 첨삭 수준 선택 ────────────────────────────────────────────────
st.markdown("")
level_choice = st.radio(
    "첨삭 수준",
    options=["fast", "full"],
    format_func=lambda x: (
        "⚡ 문법 교정 (빠름) — 맞춤법·문법 오류만"
        if x == "fast" else
        "✨ 윤문 첨삭 (정확) — 문장 구조·어휘·논리 전체"
    ),
    index=0 if st.session_state.correction_level == "fast" else 1,
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.correction_level = level_choice
sel_model = MODEL_FAST if level_choice == "fast" else MODEL_FULL

_start_tok = estimate_start_tokens(_char_cnt, level_choice, text_input) if _char_cnt else 1024
st.markdown(
    f'<div class="tok-info">모델: <strong>{sel_model}</strong> &nbsp;|&nbsp; '
    f'현재 글자 수 {_char_cnt:,}자 → 시작 토큰 <strong>{_start_tok:,}</strong> '
    f'(부족 시 자동 증가)</div>',
    unsafe_allow_html=True,
)


def _build_prompt(orig: str, level: str) -> str:
    if level == "fast":
        # ── 문법만 (lite) ─────────────────────────────────────────
        lang_ko = "ko:맞춤법·띄어쓰기·어미/문장구조/어휘"
        lang_en = "en:grammar(SVA·tense·articles·prepositions)/spelling·punctuation"
        return (
            "교정전문가.맞춤법·문법만교정.JSON만출력.\n"
            f'원문:"{orig}"\n'
            f"언어감지: {lang_ko}|{lang_en}\n"
            "출력(JSON만):\n"
            '{"lang":"ko|en","score":0-100,"summary":"1문장이내해설",'
            '"criteria":[{"type":"grammar|spelling","label":"라벨","issue":"25자이내","reason":"교정이유20자이내"}],'
            '"changes":[{"orig":"원문구절","new":"수정","type":"grammar|spelling"}]}\n'
            "규칙:criteria≤3실제문제,changes전체,각change에type필드,orig=원문정확복사,JSON만"
        )
    else:
        # ── 윤문 첨삭 (flash) — 한국어 10기준 · 영어 10기준 완전 적용 ──
        lang_ko = (
            "ko10기준:"
            "theme(주제명확성·중심주장)|comprehension(제시문이해·핵심개념파악)|"
            "logic(논리전개·근거·예시)|organization(문단구성·서론본론결론)|"
            "clarity(문장표현·주술호응·어순)|vocabulary(어휘선택·적절성·전문용어)|"
            "grammar(맞춤법·띄어쓰기·문장부호)|conditions(시험조건충족·형식·글자수)|"
            "critical(비판적사고·다각도판단·논거)|unity(완성도·통일성·톤일관성)"
        )
        lang_en = (
            "en10기준:"
            "grammar(tense·SVA·articles·prepositions·pronouns)|"
            "spelling(errors·commas·periods·apostrophes)|"
            "clarity(meaning immediately clear per sentence)|"
            "structure(word-order·sentence-length·fragments)|"
            "vocabulary(no-repetition·topic-appropriate)|"
            "logic(sentences·paragraphs connect naturally)|"
            "theme(answers question·no unnecessary content)|"
            "organization(intro·body·conclusion clear)|"
            "repetition(no overuse same words·expressions·examples)|"
            "conciseness(not verbose·key points clearly)"
        )
        types_en = "grammar|spelling|clarity|structure|vocabulary|logic|theme|organization|repetition|conciseness"
        types_ko = "theme|comprehension|logic|organization|clarity|vocabulary|grammar|conditions|critical|unity"
        return (
            "첨삭전문가.원문전체분석.JSON만출력.\n"
            f'원문:"{orig}"\n'
            f"언어감지→기준적용:\n{lang_ko}\n{lang_en}\n"
            "출력(JSON만):\n"
            f'{{"lang":"ko|en","score":0-100,"summary":"2문장이내전체해설",'
            f'"criteria":[{{"type":"{types_en}(en) or {types_ko}(ko)",'
            '"label":"라벨","issue":"30자이내","reason":"이교정이필요한이유25자이내"}}],'
            '"changes":[{"orig":"원문구절","new":"수정","type":"해당기준유형"}]}}\n'
            "규칙:en→10기준중실제문제≤5개,ko→10기준중실제문제≤5개,"
            "changes전체빠짐없이,각change에type필드필수,orig=원문정확복사,JSON만"
        )


def _parse_and_store(raw: str, orig_text: str) -> bool:
    """JSON 파싱 후 session_state 저장. 성공 True, 실패 False."""
    cleaned = re.sub(r"```json?|```", "", raw).strip()
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return False
    data = json.loads(m.group())
    st.session_state.score         = data.get("score")
    st.session_state.lang          = data.get("lang", "ko")
    st.session_state.criteria      = data.get("criteria", [])
    st.session_state.summary       = data.get("summary", "")
    changes = data.get("changes", [])
    markup, markup_html, clean = apply_changes(orig_text, changes)
    st.session_state.markup_text   = markup
    st.session_state.markup_html   = markup_html
    st.session_state.edited_text   = clean
    st.session_state.analysis_done = True
    return True


def _run_with_escalation(prompt: str, model: str, orig_text: str, level: str) -> bool:
    """
    글자 수 기반 최적 시작 토큰부터 자동 증가 재시도.
    성공 True, 8192 모두 실패 시 False (needs_extended_confirm 설정).
    """
    char_count   = _cc(orig_text)
    start_tokens = estimate_start_tokens(char_count, level, orig_text)
    start_idx    = TOKEN_LEVELS.index(start_tokens)

    for i in range(start_idx, len(TOKEN_LEVELS)):
        max_tok = TOKEN_LEVELS[i]
        try:
            raw = call_gemini(prompt, max_tokens=max_tok, model=model)
            if _parse_and_store(raw, orig_text):
                st.session_state.used_tokens            = max_tok
                st.session_state.needs_extended_confirm = False
                return True
            # JSON 파싱 실패 → 토큰 부족으로 간주
            raise ValueError("JSON 파싱 실패")
        except (json.JSONDecodeError, ValueError):
            if i < len(TOKEN_LEVELS) - 1:
                next_tok = TOKEN_LEVELS[i + 1]
                st.toast(f"⚠️ {max_tok:,} 토큰 부족 → {next_tok:,} 토큰으로 재시도 중...")
            continue
        except Exception:
            raise   # 네트워크·할당량 오류는 그대로 전파
    # 모든 레벨 실패
    st.session_state.needs_extended_confirm = True
    return False


# ── AI 첨삭 버튼 ──────────────────────────────────────────────────
if st.button(
    "🤖 AI 첨삭 시작하기 (Gemini)",
    type="primary",
    use_container_width=True,
    disabled=len(text_input.strip()) < 5,
):
    orig = text_input.strip()
    st.session_state.orig_text              = orig
    st.session_state.needs_extended_confirm = False

    with st.spinner(f"AI 첨삭 중... ({sel_model})"):
        try:
            prompt = _build_prompt(orig, level_choice)
            _run_with_escalation(prompt, sel_model, orig, level_choice)
        except Exception as e:
            st.error(f"첨삭 오류: {e}")
            st.caption("💡 API 키가 올바른지, 네트워크가 연결되어 있는지 확인해주세요.")
            st.stop()

    st.rerun()

# ── 8192 초과 시 확인 대화 ─────────────────────────────────────────
if st.session_state.get("needs_extended_confirm"):
    st.warning(
        f"⚠️ 8,192 토큰으로도 처리를 완료하지 못했습니다.\n\n"
        f"글이 매우 길거나 변경 사항이 많습니다. "
        f"최대 {EXTENDED_TOKENS:,} 토큰으로 계속 진행하시겠습니까?"
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ 계속 진행", type="primary", use_container_width=True):
            orig = st.session_state.orig_text
            with st.spinner(f"AI 첨삭 중... (최대 {EXTENDED_TOKENS:,} 토큰)"):
                try:
                    prompt = _build_prompt(orig, st.session_state.correction_level)
                    raw    = call_gemini(prompt, max_tokens=EXTENDED_TOKENS, model=sel_model)
                    if _parse_and_store(raw, orig):
                        st.session_state.used_tokens            = EXTENDED_TOKENS
                        st.session_state.needs_extended_confirm = False
                    else:
                        st.error("최대 토큰으로도 처리 실패. 글을 나눠서 첨삭해주세요.")
                        st.session_state.needs_extended_confirm = False
                        st.stop()
                except Exception as e:
                    st.error(f"첨삭 오류: {e}")
                    st.stop()
            st.rerun()
    with col_no:
        if st.button("❌ 취소", use_container_width=True):
            st.session_state.needs_extended_confirm = False
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

    # ── 첨삭 해설 ─────────────────────────────────────────────────
    if st.session_state.get("summary"):
        used_tok = st.session_state.get("used_tokens")
        tok_info = f" &nbsp;|&nbsp; 사용 토큰: {used_tok:,}" if used_tok else ""
        level_lbl = "⚡ 문법 교정" if st.session_state.correction_level == "fast" else "✨ 윤문 첨삭"
        st.markdown(
            f'<div style="background:#eef2ff;border-left:4px solid #4f46e5;'
            f'border-radius:10px;padding:14px 18px;margin-bottom:12px;font-size:14px;line-height:1.8">'
            f'<div style="font-size:11px;color:#6b7280;margin-bottom:6px">'
            f'📝 첨삭 해설 &nbsp;|&nbsp; {level_lbl}{tok_info}</div>'
            f'{esc(st.session_state.summary)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander(f"📋 첨삭 기준 ({lang_badge})", expanded=True):
        render_criteria(st.session_state.criteria, lang)

    tab_cmp, tab_final = st.tabs(["📄 원문 vs 첨삭본", "✨ 최종 완성본"])

    with tab_cmp:
        # 원문(좌) | 첨삭본(우) 나란히 비교
        st.markdown(
            '<div class="legend">'
            '색상은 첨삭 기준 유형과 동일 &nbsp;|&nbsp; '
            '<span style="text-decoration:line-through;opacity:.7">취소선</span> 삭제 &nbsp;&nbsp;'
            '<strong>굵게</strong> 수정 내용</div>',
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
            _mhtml = st.session_state.get("markup_html") or render_markup(st.session_state.markup_text)
            st.markdown(
                f'<div class="markup-box" style="min-height:200px">{_mhtml}</div>',
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
