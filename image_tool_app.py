import io
import math
import os
import re
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageChops

# 폰트 크기 확대 및 스타일링 적용 (제목 제외 전체 폰트 업)
st.markdown("""
    <style>
    p, label, .streamlit-expanderHeader, div[data-baseweb="checkbox"] span, div[data-baseweb="slider"] div {
        font-size: 18px !important;
    }
    .stDownloadButton button, .stButton button {
        font-size: 18px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="오픈마켓 상품 이미지 맞춤 가공 툴", page_icon="🖼️", layout="centered")


# =========================================================
# 비밀번호 로그인 (배포 후 아무나 URL로 접속하지 못하도록)
# =========================================================

def _password_from_url():
    """
    주소 뒤에 붙은 ?pw=... 값을 읽습니다.

    ⚠️ Streamlit 버전에 따라 쿼리 파라미터 API가 다릅니다
       (1.30 이후 st.query_params / 그 이전 st.experimental_get_query_params).
       배포 환경 버전에 따라 한쪽만 있을 수 있어 둘 다 시도합니다.
    """
    try:
        val = st.query_params.get("pw", "")
        if isinstance(val, (list, tuple)):
            val = val[0] if val else ""
        return str(val or "")
    except Exception:
        pass
    try:
        vals = st.experimental_get_query_params().get("pw") or []
        return str(vals[0]) if vals else ""
    except Exception:
        return ""


def check_password():
    """secrets.toml(로컬) 또는 Streamlit Cloud의 Secrets 설정에 저장된 비밀번호와 비교.
    맞으면 True를 반환하고, 이후 세션 동안은 다시 묻지 않음.

    ⚠️ 소싱도구에서 '이미지가공 툴 열기'로 넘어올 때는 주소에 ?pw=... 가
       붙어 있습니다. 예전에는 이 값을 읽는 코드가 아예 없어서, 링크로
       들어와도 매번 비밀번호를 다시 물었습니다. 이제 URL의 값이 맞으면
       바로 통과시킵니다.
       (비밀번호가 주소에 노출되므로 브라우저 기록에 남습니다 — 외부 유출이
        곤란한 자료를 다루는 툴이 아니라 편의를 택했습니다.)
    """

    def password_entered():
        if str(st.session_state.get("password", "")) == str(st.secrets.get("password", "")):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    # 1) 주소로 넘어온 비밀번호로 자동 로그인
    url_pw = _password_from_url()
    if url_pw and url_pw == str(st.secrets.get("password", "")):
        st.session_state["password_correct"] = True
        return True

    # 2) 안 맞거나 없으면 직접 입력
    st.title("🔒 로그인")
    st.text_input("비밀번호를 입력하세요", type="password", on_change=password_entered, key="password")
    if st.session_state.get("password_correct") is False:
        st.error("비밀번호가 올바르지 않습니다.")
    elif url_pw:
        st.warning("주소에 붙은 비밀번호가 맞지 않습니다. 직접 입력해주세요.", icon="⚠️")
    return False


if not check_password():
    st.stop()


st.title("🖼️ 오픈마켓 상품 이미지 맞춤 가공 툴")
st.markdown("미러 반사, 지정 수량 포개기, 파스텔/다크 스튜디오 배경, 원근감 포개기 효과를 제공합니다.")


# =========================================================
# 이미지 가공 함수
# =========================================================

def normalize_to_rgb(orig_img):
    """투명 배경(RGBA/LA/팔레트 투명)은 흰 배경으로 합성, 그 외는 RGB로 변환"""
    if orig_img.mode in ("RGBA", "LA") or (orig_img.mode == "P" and "transparency" in orig_img.info):
        rgba_img = orig_img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        return Image.alpha_composite(white_bg, rgba_img).convert("RGB")
    return orig_img.convert("RGB")


# =========================================================
# 오픈마켓 출력 규격 맞추기
# =========================================================
# ⚠️ 오픈마켓 대표이미지는 정사각형 1,000x1,000이 표준입니다(쿠팡 기준:
#    권장 1,000x1,000 / 최소 500px 이상 / 10MB 이하 JPG·PNG).
#    규격에 못 미치면 마켓이 "이미지를 추출할 수 없습니다. 올바른 URL 주소를
#    입력해주세요"라는 엉뚱한 메시지로 거절해서, URL이 잘못된 줄 알고 한참
#    헤매게 됩니다(실제로 441x775 결과물로 겪은 일).
#
#    효과마다 캔버스 크기가 제각각(반사는 세로로 길고, 포개기는 가로로 넓음)
#    이라 효과별로 맞추면 관리가 어렵습니다. 그래서 저장 직전에 한 곳에서만
#    규격을 맞춥니다 — 나중에 효과를 추가해도 자동으로 적용됩니다.

MARKET_SIZE_OPTIONS = {
    "1000 x 1000 (오픈마켓 표준)": 1000,
    "1200 x 1200 (조금 더 크게)": 1200,
    "원본 크기 유지 (규격 안 맞춤)": 0,
}


def fit_to_square(im, size=1000, bg=(255, 255, 255)):
    """
    비율을 유지한 채 가장 긴 변을 size에 맞추고, 남는 여백은 흰색으로 채워
    정사각형으로 만듭니다.

    반환: (정사각형 이미지, 확대되었는지 여부)
      - 확대 여부를 같이 돌려주는 이유: 원본이 작으면 없는 픽셀을 지어내는
        확대가 일어나 화질이 뭉개집니다. 이건 코드로 해결할 수 없고 더 큰
        원본으로 다시 찍는 수밖에 없어서, 사용자에게 알려줘야 합니다.
    """
    im = im.convert("RGB")
    w, h = im.size
    if size <= 0:
        return im, False
    if (w, h) == (size, size):
        return im, False

    ratio = min(size / w, size / h)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    resized = im.resize((new_w, new_h), resample)

    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas, ratio > 1.0


@st.cache_data(show_spinner=False)
def remove_light_background(img_rgb, threshold=240, feather=2, downsample=300, refine_iters=60):
    """흰색/밝은 배경을 투명 처리. 가장자리에서 연결된 배경 영역만 배경으로 인식(flood fill)해
    제품 안쪽의 흰 글자/로고는 보존합니다. 저해상도로 먼 거리 전파를 빠르게 끝낸 뒤, 원본
    해상도에서 소량만 정밀 보정하는 2단계 방식이라 - 속도도 빠르고 경계도 매끄럽습니다."""
    w, h = img_rgb.size
    gray_full = np.array(img_rgb.convert("L"))
    is_bg_full = gray_full >= threshold

    def grow(reached, is_bg, max_iter):
        for _ in range(max_iter):
            grown = reached.copy()
            grown[1:, :] |= reached[:-1, :]
            grown[:-1, :] |= reached[1:, :]
            grown[:, 1:] |= reached[:, :-1]
            grown[:, :-1] |= reached[:, 1:]
            grown &= is_bg
            if np.array_equal(grown, reached):
                return grown, True
            reached = grown
        return reached, False

    # 1) 저해상도로 먼 거리까지의 배경 연결성을 빠르게 계산 (큰 이미지에서도 빠르게 수렴)
    scale = min(1.0, downsample / max(w, h))
    if scale < 1.0:
        small = img_rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        is_bg_small = np.array(small.convert("L")) >= threshold
        reached_small = np.zeros_like(is_bg_small, dtype=bool)
        reached_small[0, :] = is_bg_small[0, :]
        reached_small[-1, :] = is_bg_small[-1, :]
        reached_small[:, 0] = is_bg_small[:, 0]
        reached_small[:, -1] = is_bg_small[:, -1]
        reached_small, _ = grow(reached_small, is_bg_small, 500)
        seed_full = np.array(
            Image.fromarray((reached_small * 255).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
        ) > 127
        reached = seed_full & is_bg_full
    else:
        reached = np.zeros_like(is_bg_full, dtype=bool)

    # 2) 이미지 테두리는 항상 시드로 포함
    reached[0, :] |= is_bg_full[0, :]
    reached[-1, :] |= is_bg_full[-1, :]
    reached[:, 0] |= is_bg_full[:, 0]
    reached[:, -1] |= is_bg_full[:, -1]

    # 3) 원본 해상도에서 소량만 정밀 보정 (저해상도 결과가 대부분을 이미 커버하므로 반복 횟수가 적어도 됨)
    reached, _ = grow(reached, is_bg_full, refine_iters)

    fg_mask = Image.fromarray((~reached * 255).astype(np.uint8), mode="L")  # 255=제품, 0=배경
    fg_mask = fg_mask.filter(ImageFilter.MedianFilter(5))  # 압축 노이즈 등으로 인한 잔 톱니 제거
    # 흰색 경계가 섞인 안티에일리어싱 픽셀이 반투명 헤일로로 남지 않도록 살짝 침식
    fg_mask = fg_mask.filter(ImageFilter.MinFilter(3))
    fg_mask = fg_mask.filter(ImageFilter.MinFilter(3))

    if feather > 0:
        fg_mask = fg_mask.filter(ImageFilter.GaussianBlur(feather))

    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(fg_mask)
    return rgba


def get_content_bbox(img_rgb, bg_threshold=200):
    """실제 제품 영역 bbox. 실제 상품 사진에는 스튜디오 촬영 시 생긴 은은한 사전 그림자가
    이미 깔려 있는 경우가 많은데, 임계값이 너무 느슨하면(예: 245) 그 옅은 그림자까지
    '제품'으로 잡혀서 bbox가 실제 제품보다 아래로 늘어나 버림 (반사/그림자 위치가 밀리는 원인).
    기본값을 더 엄격하게 잡아서 이런 사전 그림자는 배경으로 보고 제외함."""
    gray = img_rgb.convert("L")
    mask = gray.point(lambda p: 255 if p < bg_threshold else 0)
    bbox = mask.getbbox()
    return bbox if bbox else (0, 0, img_rgb.width, img_rgb.height)


def tighten_to_content(img_rgb, margin_ratio=0.04, bg_threshold=200):
    """원본 사진의 여백을 제거해 제품에 딱 맞게 크롭 (그림자·반사 위치 계산 기준을 맞추기 위함)"""
    l, t, r, b = get_content_bbox(img_rgb, bg_threshold)
    mw = int((r - l) * margin_ratio)
    mh = int((b - t) * margin_ratio)
    l2, t2 = max(0, l - mw), max(0, t - mh)
    r2, b2 = min(img_rgb.width, r + mw), min(img_rgb.height, b + mh)
    if r2 <= l2 or b2 <= t2:
        return img_rgb
    return img_rgb.crop((l2, t2, r2, b2))


def parse_count(text, default=6, max_count=60):
    digits = re.sub(r"[^0-9]", "", str(text))
    if not digits:
        return default
    return max(1, min(max_count, int(digits)))


def get_lightest_dominant_color(img_rgb, min_ratio=0.01):
    """배경(흰색 계열)을 제외하고, 픽셀 비중이 일정 이상인 색상 중 가장 밝은 색을 반환"""
    small = img_rgb.copy()
    small.thumbnail((120, 120))
    colors = small.getcolors(small.width * small.height)
    if not colors:
        return (245, 240, 230)
    total = small.width * small.height
    candidates = [c for c in colors if c[0] / total >= min_ratio]
    candidates = [c for c in candidates if not (c[1][0] > 238 and c[1][1] > 238 and c[1][2] > 238)]
    if not candidates:
        candidates = colors
    candidates.sort(key=lambda c: sum(c[1][:3]), reverse=True)
    return candidates[0][1]


def lighten_color(rgb, amount=0.82):
    r, g, b = rgb[:3]
    return (
        int(r + (255 - r) * amount),
        int(g + (255 - g) * amount),
        int(b + (255 - b) * amount),
    )


def create_angled_shadow(width, height, angle_deg=15, blur=16, opacity=100, ellipse_h_ratio=0.16, shape="oval"):
    """지정 너비에 맞는, 우측 하단으로 기울어진 부드러운 그림자 레이어와 x오프셋 반환.
    shape="oval": 둥근 제품(병 등)에 어울리는 타원형. shape="flat": 각진 제품(박스 등)에
    어울리는, 모서리가 살짝 둥근 납작한 사각형 그림자."""
    pad = blur * 2
    shadow_h = max(16, int(height * ellipse_h_ratio))
    layer = Image.new("RGBA", (width + pad * 2, shadow_h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box = [pad, pad, pad + width, pad + shadow_h]
    if shape == "flat":
        radius = max(2, int(shadow_h * 0.35))
        draw.rounded_rectangle(box, radius=radius, fill=(25, 20, 15, opacity))
    else:
        draw.ellipse(box, fill=(25, 20, 15, opacity))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    dx = int(math.sin(math.radians(angle_deg)) * height * 0.35)
    return layer, dx, pad


# ---------- 1. 미러 반사 효과 (고려은단 스타일) ----------

def apply_mirror_reflection(original, reflect_ratio=0.28, max_opacity=90, falloff=1.8):
    """제품 바로 밑에 딱 붙어서 반사되는 거울 반사. falloff를 1보다 크게 주면 반사가 상단(제품과
    맞닿는 지점)에서는 진하고 아래로 갈수록 빠르게 옅어져서, 바닥에 딱 붙은 느낌이 남."""
    w, h = original.size
    bbox = get_content_bbox(original)
    content_bottom = bbox[3]

    base = original.crop((0, 0, w, content_bottom))
    strip_h = max(10, int((content_bottom - bbox[1]) * reflect_ratio))
    strip_h = min(strip_h, content_bottom)
    strip = original.crop((0, content_bottom - strip_h, w, content_bottom))

    flipped = strip.transpose(Image.FLIP_TOP_BOTTOM)
    y = np.arange(strip_h, dtype=np.float32).reshape(strip_h, 1)
    fade = np.clip(1 - y / strip_h, 0, 1) ** falloff
    alpha_arr = np.broadcast_to((fade * max_opacity).astype(np.uint8), (strip_h, w))
    alpha = Image.fromarray(alpha_arr, mode="L")
    flipped.putalpha(alpha)

    margin = int(strip_h * 0.25)
    canvas = Image.new("RGB", (w, content_bottom + strip_h + margin), (255, 255, 255))
    canvas.paste(base, (0, 0))
    canvas.paste(flipped.convert("RGB"), (0, content_bottom), flipped)
    return canvas


# ---------- 2. 지정 수량 포개기 (우상향 대각선) + 각도 그림자 ----------

def apply_stack_fan(tight_img, count, cols=6, overlap_pct=70, angle_deg=15, blur=16, rise_ratio=0.05, scale_step=0.96):
    """tight_img는 tighten_to_content()로 여백을 제거한 이미지여야 그림자가 올바르게 보입니다.
    맨 앞(왼쪽 아래)이 완전히 보이고, 뒤로 갈수록 우상향으로 겹쳐 올라가는 삼다수 스타일 배치.
    뒤로 갈수록 살짝 작아지게(scale_step) 해서 단순 평행이동보다 입체감이 나도록 함."""
    w, h = tight_img.size
    cutout = remove_light_background(tight_img)  # 실제 배경만 투명 → 뒤 제품의 진짜 여백 부분만 겹침 영향 최소화
    cols = max(1, min(cols, count))
    rows = math.ceil(count / cols)

    step_x = max(int(w * (1 - overlap_pct / 100.0)), int(w * 0.08))
    step_y = max(1, int(h * rise_ratio))
    max_rise = step_y * (cols - 1)
    row_gap = max(int(h * 0.14), blur * 3)
    pad = 30

    slot_h = h + max_rise
    canvas_w = w + step_x * (cols - 1) + pad * 2
    canvas_h = pad * 2 + slot_h * rows + row_gap * max(0, rows - 1) + blur * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    idx = 0
    for r in range(rows):
        this_row_cols = min(cols, count - idx)
        row_top = pad + r * (slot_h + row_gap)
        row_base_bottom = row_top + max_rise + h  # 맨 앞(왼쪽, depth 0) 제품의 바닥선(y좌표)
        row_base_top = row_base_bottom - h  # 맨 앞(depth 0) 제품의 top y좌표
        row_width = w + step_x * (this_row_cols - 1)

        # 그림자는 제품 바닥선에 딱 맞닿게만 배치 (제품 안쪽으로 파고들지 않게 해서, 어떤 제품이 위에
        # 덮이더라도 그림자가 지워지지 않도록 함)
        shadow, dx, spad = create_angled_shadow(row_width, h, angle_deg=angle_deg, blur=blur)
        shadow_x = pad - spad + dx
        shadow_y = row_base_bottom - spad
        canvas.paste(shadow, (shadow_x, shadow_y), shadow)

        # 뒤(오른쪽 위, 작게)부터 그린 뒤 앞(왼쪽 아래, 원래 크기)을 마지막에 그려서 맨 앞이 완전히 보이게 함
        # top 기준으로 위치를 잡아야, 축소 비율과 상관없이 뒤로 갈수록 top과 bottom이 모두 확실히
        # 위로 올라감 (bottom만 기준으로 하면 축소폭이 상승폭보다 클 때 오히려 내려가 보이는 문제가 있었음)
        for c in range(this_row_cols - 1, -1, -1):
            scale = scale_step ** c
            item = cutout if scale >= 0.999 else cutout.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            item_top = row_base_top - c * step_y  # 뒤로 갈수록 top이 항상 위로 올라감 (보장됨)
            x = pad + c * step_x
            canvas.paste(item, (x, item_top), item)
        idx += this_row_cols
    return canvas


# ---------- 3. 제품 색상 기반 파스텔 배경 + 그림자 ----------

def apply_pastel_bg(tight_img, width_ratio=1.35, height_ratio=1.3, angle_deg=15, blur=14, lighten_amount=0.82,
                     shadow_shape="oval"):
    w, h = tight_img.size
    bbox = get_content_bbox(tight_img)
    content_h = max(1, bbox[3] - bbox[1])
    base_color = get_lightest_dominant_color(tight_img)
    pastel = lighten_color(base_color, amount=lighten_amount)
    pastel_deep = lighten_color(base_color, amount=max(0.45, lighten_amount - 0.14))

    can_w, can_h = int(w * width_ratio), int(h * height_ratio)
    bg = Image.new("RGB", (can_w, can_h), pastel)
    draw = ImageDraw.Draw(bg)
    for y in range(can_h):
        ratio = y / can_h
        rr = int(pastel[0] + (pastel_deep[0] - pastel[0]) * ratio)
        gg = int(pastel[1] + (pastel_deep[1] - pastel[1]) * ratio)
        bb = int(pastel[2] + (pastel_deep[2] - pastel[2]) * ratio)
        draw.line([(0, y), (can_w, y)], fill=(rr, gg, bb))

    pos_x = (can_w - w) // 2
    pos_y = int(can_h * 0.08)

    # 각진(사각) 제품은 납작하고 각진 그림자, 둥근 제품은 부드러운 타원 그림자
    h_ratio = 0.11 if shadow_shape == "flat" else 0.16
    shadow, dx, spad = create_angled_shadow(w, content_h, angle_deg=angle_deg, blur=blur, opacity=90,
                                             ellipse_h_ratio=h_ratio, shape=shadow_shape)
    shadow_x = pos_x - spad + dx
    shadow_y = pos_y + bbox[3] - int(content_h * 0.08) - spad
    bg.paste(shadow, (shadow_x, shadow_y), shadow)

    cutout = remove_light_background(tight_img)
    bg.paste(cutout, (pos_x, pos_y), cutout)
    return bg


# ---------- 4. 스튜디오 다크 그라디언트 + 나란히 배치 + 하단 반사 (콘드로이친 스타일) ----------

def arrange_side_by_side(tight_img, count, gap_ratio=0.08):
    """흰 배경 위에 동일 이미지를 겹치지 않게 나란히 배치 (Wonfit 스타일)"""
    w, h = tight_img.size
    if count <= 1:
        return tight_img
    gap = max(1, int(w * gap_ratio))
    canvas_w = w * count + gap * (count - 1)
    canvas = Image.new("RGB", (canvas_w, h), (255, 255, 255))
    for i in range(count):
        canvas.paste(tight_img, (i * (w + gap), 0))
    return canvas


def make_dark_gradient_bg(size, wall_color=(178, 180, 184), floor_color=(55, 57, 62),
                           horizon_ratio=0.56, transition=0.22, grain=5):
    """센트룸 스타일 투톤 스튜디오 배경: 위쪽 벽은 밝게, 아래쪽 바닥은 진하게,
    그 사이를 부드러운 커브(smoothstep)로 자연스럽게 전환. numpy로 벡터 연산."""
    w, h = size
    y = np.arange(h, dtype=np.float32).reshape(h, 1)
    horizon_y = h * horizon_ratio
    trans_h = max(1.0, h * transition)
    t = np.clip((y - (horizon_y - trans_h / 2)) / trans_h, 0.0, 1.0)
    t = t * t * (3 - 2 * t)  # smoothstep - 급격한 경계선 없이 부드럽게 벽→바닥 전환

    wall = np.array(wall_color, dtype=np.float32)
    floor = np.array(floor_color, dtype=np.float32)
    row_color = wall * (1 - t) + floor * t  # (h, 3)
    arr = np.broadcast_to(row_color.reshape(h, 1, 3), (h, w, 3)).copy()

    if grain > 0:
        noise = np.random.randint(-grain, grain + 1, size=(h, w, 1), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def apply_studio_dark_bg(tight_img, count=3, gap_ratio=0.10, reflect_ratio=0.32, reflect_opacity=60,
                          wall_bright=178, floor_bright=55, horizon_ratio=0.56, grain=5):
    row = arrange_side_by_side(tight_img, count, gap_ratio)
    w, h = row.size
    can_w, can_h = int(w * 1.22), int(h * 1.55)
    wall_c = (wall_bright, wall_bright + 2, wall_bright + 6)
    floor_c = (floor_bright, floor_bright + 2, floor_bright + 7)
    bg = make_dark_gradient_bg((can_w, can_h), wall_c, floor_c, horizon_ratio, 0.22, grain)

    # 은은한 스포트라이트 글로우 (벽 쪽, 제품 뒤로 살짝)
    glow = Image.new("L", (can_w, can_h), 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse([int(can_w * 0.12), int(can_h * 0.0), int(can_w * 0.88), int(can_h * horizon_ratio * 0.9)], fill=55)
    glow = glow.filter(ImageFilter.GaussianBlur(int(can_w * 0.08)))
    light_layer = Image.new("RGB", (can_w, can_h), (min(255, wall_bright + 30),) * 3)
    bg = Image.composite(light_layer, bg, glow)

    pos_x = (can_w - w) // 2
    pos_y = int(can_h * 0.05)

    cutout = remove_light_background(row)
    bg.paste(cutout, (pos_x, pos_y), cutout)

    # 하단 반사 (합성된 전체 줄 기준)
    bbox = get_content_bbox(row)
    content_bottom = bbox[3]
    strip_h = max(10, int((content_bottom - bbox[1]) * reflect_ratio))
    strip_h = min(strip_h, content_bottom)
    strip_cut = remove_light_background(row.crop((0, content_bottom - strip_h, w, content_bottom)))
    flipped = strip_cut.transpose(Image.FLIP_TOP_BOTTOM)

    fade = Image.new("L", (w, strip_h), 0)
    fd = ImageDraw.Draw(fade)
    for y in range(strip_h):
        fd.line([(0, y), (w, y)], fill=int(reflect_opacity * (1 - y / strip_h)))
    orig_alpha = flipped.split()[3]
    combined_alpha = ImageChops.multiply(orig_alpha, fade)
    flipped.putalpha(combined_alpha)

    bg.paste(flipped, (pos_x, pos_y + content_bottom), flipped)
    return bg


# ---------- 5. 원근감 있는 앞뒤 포개기 (베지밀 스타일, 중앙+좌우 배치, 각 이미지 60~70%+ 노출) ----------

def apply_perspective_fan(tight_img, count=3, rotate_deg=0, scale_step=0.92, dx_ratio=0.60, dy_ratio=0.16):
    """맨 앞 제품을 중앙에 두고, 나머지를 좌/우로 번갈아 뒤쪽(위+작게)에 배치.
    회전을 주지 않아(기본값 0) 모든 제품이 바닥에 안정적으로 서 있는 것처럼 보입니다."""
    w, h = tight_img.size
    cutout = remove_light_background(tight_img)

    dx_step = int(w * dx_ratio)
    dy_step = int(h * dy_ratio)

    # 레이어 배치: 0=정면 중앙, 이후 홀수=오른쪽 뒤, 짝수=왼쪽 뒤 (깊이 = 몇 번째 뒤 줄인지)
    layers = [(0, 0, 0)]
    i = 1
    while len(layers) < count:
        depth = (i + 1) // 2
        side = 1 if i % 2 == 1 else -1
        layers.append((side * dx_step * depth, dy_step * depth, depth))
        i += 1

    max_dx = max(abs(l[0]) for l in layers)
    max_dy = max(l[1] for l in layers)
    pad = int(max(w, h) * 0.15)

    canvas_w = w + max_dx * 2 + pad * 2
    canvas_h = h + max_dy + pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    center_x = canvas_w // 2
    bottom_y = canvas_h - pad  # 맨 앞(정면) 제품의 바닥선 - 모든 제품이 이 선을 기준으로 '뒤로 갈수록 위로' 배치됨

    # 뒤(깊이 큰 것)부터 그려서 앞(정면, 깊이0)이 마지막에 그려지게 함
    for dx, dy, depth in sorted(layers, key=lambda t: -t[2]):
        scale = scale_step ** depth
        item = cutout.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        if rotate_deg and depth > 0:
            angle = rotate_deg * depth * (1 if dx > 0 else -1)
            item = item.rotate(angle, expand=True, resample=Image.BICUBIC)

        item_bottom = bottom_y - dy  # 깊이가 클수록(뒤로 갈수록) 바닥선이 위로 올라감 = 원근감
        x = center_x + dx - item.width // 2
        y = item_bottom - item.height
        canvas.paste(item, (x, y), item)

    return canvas


# =========================================================
# 사이드바 - 옵션
# =========================================================

st.sidebar.header("⚙️ 가공 옵션 선택")

st.sidebar.subheader("1. 미러 반사효과")
opt_reflection = st.sidebar.checkbox("반사 효과 적용", value=True, key="opt_reflection")
reflect_count = st.sidebar.number_input("나란히 개수 (1=단일 제품, 2 이상=제품을 나란히 배치)", min_value=1, max_value=8, value=1, key="reflect_count")
reflect_gap = st.sidebar.slider("나란히 배치 간격 (%)", 2, 30, 2, key="reflect_gap")
reflect_ratio = st.sidebar.slider("반사 높이 비율", 0.10, 0.6, 0.28, 0.02, key="reflect_ratio")
reflect_opacity = st.sidebar.slider("반사 최대 불투명도", 30, 150, 90, 5, key="reflect_opacity")
reflect_falloff = st.sidebar.slider("반사 감쇠 속도 (클수록 바닥에 딱 붙어 빨리 옅어짐)", 1.0, 3.0, 1.8, 0.1, key="reflect_falloff")

st.sidebar.subheader("2. 포개기 효과")
opt_stack = st.sidebar.checkbox("포개기 효과 적용", value=True, key="opt_stack")
default_stack_count = st.sidebar.text_input(
    "기본 포개기 개수 (이미지 업로드 시 각 이미지에 기본값으로 적용, 이미지별로 개별 수정 가능)",
    value="3", key="default_stack_count"
)
stack_cols = st.sidebar.slider("한 줄 최대 개수", 2, 30, 15, key="stack_cols")
stack_overlap = st.sidebar.slider("겹침 정도 (%)", 30, 95, 40, 5, key="stack_overlap")
stack_rise = st.sidebar.slider("우상향 기울기 (%)", 0, 15, 8, key="stack_rise")
stack_depth_scale = st.sidebar.slider("뒤로 갈수록 축소 비율 (%, 입체감)", 85, 100, 94, key="stack_depth_scale")
stack_angle = st.sidebar.slider("그림자 각도 (도)", 0, 45, 15, key="stack_angle")

st.sidebar.subheader("3. 파스텔 배경")
opt_pastel = st.sidebar.checkbox("파스텔 배경 적용", value=True, key="opt_pastel")
pastel_shape = st.sidebar.radio("제품 형태 (그림자 스타일)", ["사각 (각진 그림자)", "둥근 (부드러운 타원 그림자)"], key="pastel_shape")
pastel_shadow_shape = "flat" if pastel_shape.startswith("사각") else "oval"
pastel_lighten = st.sidebar.slider("배경 밝기 (연한 정도, %)", 50, 95, 82, 1, key="pastel_lighten")
pastel_size = st.sidebar.slider("배경 여유 공간 비율", 1.10, 1.80, 1.35, 0.05, key="pastel_size")
pastel_angle = st.sidebar.slider("그림자 각도 (도) ", 0, 45, 15, key="pastel_angle")

st.sidebar.subheader("4. 스튜디오톤 배경")
opt_dark = st.sidebar.checkbox("스튜디오 그라디언트 배경 적용", value=True, key="opt_dark")
dark_count = st.sidebar.number_input("나란히 개수", min_value=1, max_value=6, value=3, key="dark_count")
dark_gap = st.sidebar.slider("배치 간격 (%)", 2, 30, 3, key="dark_gap")
dark_tone = st.sidebar.radio(
    "배경 톤", ["다크 그레이", "라이트 그레이"],
    key="dark_tone"
)
_dark_tone_defaults = {
    "다크 그레이": (178, 84),
    "라이트 그레이": (247, 205),
}
_default_wall, _default_floor = _dark_tone_defaults[dark_tone]
# 톤 프리셋마다 슬라이더 키를 다르게 둬서, 톤을 바꾸면 그 톤에 맞는 기본값으로 보이고
# 각 톤별로 조정한 값은 따로 기억됩니다.
dark_wall_bright = st.sidebar.slider("벽 밝기 (위쪽, 연하게)", 100, 250, _default_wall, key=f"dark_wall_{dark_tone}")
dark_floor_bright = st.sidebar.slider("바닥 밝기 (아래쪽, 진하게)", 20, 230, _default_floor, key=f"dark_floor_{dark_tone}")
dark_horizon = st.sidebar.slider("바닥이 시작되는 위치 (%)", 30, 80, 56, key="dark_horizon")
dark_grain = st.sidebar.slider("배경 질감(그레인) 강도", 0, 15, 5, key="dark_grain")
dark_reflect_opacity = st.sidebar.slider("바닥 반사 불투명도", 0, 150, 60, 5, key="dark_reflect_opacity")

st.sidebar.subheader("5. 원근감 포개기")
opt_perspective = st.sidebar.checkbox("원근 포개기 적용", value=True, key="opt_perspective")
persp_count = st.sidebar.number_input("포개기 개수", min_value=2, max_value=7, value=3, key="persp_count")
persp_rotate = 0  # 기울임 없이 항상 똑바로 세워진 형태로 고정
persp_dx = st.sidebar.slider("가로 이동 (%, 노출도 조절)", 30, 75, 60, key="persp_dx")
persp_dy = st.sidebar.slider("세로 이동 (%, 원근감)", 0, 35, 16, key="persp_dy")
persp_scale = st.sidebar.slider("뒤 제품 축소 비율 (%)", 80, 100, 92, key="persp_scale")

st.sidebar.subheader("기타 옵션")
opt_tight_double = st.sidebar.checkbox("상품 2개 초밀착 나란히 배치", value=False, key="opt_tight_double")

st.sidebar.subheader("📐 저장 규격")
market_size_label = st.sidebar.selectbox(
    "출력 크기", list(MARKET_SIZE_OPTIONS.keys()), index=0, key="market_size",
    help="오픈마켓 대표이미지는 정사각형 1,000x1,000이 표준입니다. "
         "규격에 못 미치면 마켓이 '올바른 URL을 입력해주세요'라는 엉뚱한 "
         "메시지로 거절합니다.",
)
market_size = MARKET_SIZE_OPTIONS[market_size_label]


# =========================================================
# 미리보기 (첫 번째 업로드 이미지로 슬라이더 값을 즉시 확인)
# =========================================================

uploaded_files = st.file_uploader("상품 이미지 파일을 선택하세요 (다중 선택 가능)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)


# =========================================================
# 소싱도구에서 넘어온 이미지 (?img=<주소>)
# =========================================================
# 소싱도구가 상품 페이지에서 가장 큰 원본을 찾아 그 주소를 넘겨줍니다.
# 내려받아 다시 올리는 왕복을 없애려고 주소를 직접 받아 처리합니다.
# ⚠️ 이 앱은 Streamlit Cloud(외부 서버)에서 돌기 때문에, 상품 사이트가
#    외부 접근을 막아두면 못 받아올 수 있습니다. 그때는 실패를 감추지 않고
#    알려주고, 평소처럼 파일 업로드로 진행하시게 안내합니다.

class _UrlImage(io.BytesIO):
    """업로드된 파일처럼 쓰기 위한 래퍼 (Image.open과 .name 둘 다 필요)."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _image_from_url_param():
    try:
        val = st.query_params.get("img", "")
        if isinstance(val, (list, tuple)):
            val = val[0] if val else ""
    except Exception:
        try:
            vals = st.experimental_get_query_params().get("img") or []
            val = vals[0] if vals else ""
        except Exception:
            val = ""
    url = str(val or "").strip()
    if not url:
        return None, ""

    cache = st.session_state.setdefault("_url_img_cache", {})
    if url in cache:
        data, err = cache[url]
    else:
        try:
            import requests
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data, err = resp.content, ""
        except Exception as e:
            data, err = None, f"{type(e).__name__}: {e}"
        cache[url] = (data, err)

    if not data:
        return None, err
    name = os.path.basename(url.split("?")[0]) or "sourced.jpg"
    if not re.search(r"\.(png|jpe?g|webp)$", name, re.I):
        name += ".jpg"
    return _UrlImage(data, name), ""


_url_img, _url_img_err = _image_from_url_param()
if _url_img is not None and not uploaded_files:
    uploaded_files = [_url_img]
    try:
        _w, _h = Image.open(_url_img).size
        _url_img.seek(0)
        st.success(
            f"소싱도구에서 넘어온 원본을 불러왔습니다 — **{_w}×{_h}px** "
            f"({_url_img.name}). 아래에서 바로 가공하시면 됩니다.", icon="✅")
    except Exception:
        st.success("소싱도구에서 넘어온 이미지를 불러왔습니다.", icon="✅")
elif _url_img_err:
    st.warning(
        f"소싱도구에서 넘어온 주소로 이미지를 받지 못했습니다 ({_url_img_err}). "
        "상품 사이트가 외부 서버의 접근을 막는 경우입니다 — 그 주소를 "
        "브라우저에서 직접 열어 저장하신 뒤, 위에서 파일로 올려주세요.",
        icon="⚠️")

if uploaded_files:
    file_count = len(uploaded_files)
    st.write(f"총 **{file_count}**개의 파일이 업로드되었습니다.")

    # 이미지별 포개기 개수 지정 (기본값은 사이드바 값, 이미지별로 덮어쓰기 가능)
    stack_counts = []
    if opt_stack:
        st.markdown("#### 📦 이미지별 포개기 개수")
        st.caption("※ 최대 12개까지 포갤 수 있습니다.")
        ui_cols = st.columns(3)
        for i, f in enumerate(uploaded_files):
            with ui_cols[i % 3]:
                val = st.text_input(f"{f.name}", value=default_stack_count, key=f"stack_count_{i}")
                stack_counts.append(val)

    # 원본 해상도 점검 — 작으면 결과물이 확대되어 뭉개지므로 미리 알려줍니다.
    if market_size > 0:
        _small_srcs = []
        for f in uploaded_files:
            try:
                _w, _h = Image.open(f).size
                f.seek(0)  # 아래에서 다시 읽어야 하므로 포인터를 되돌립니다
                if min(_w, _h) < market_size:
                    _small_srcs.append(f"{f.name} ({_w}×{_h})")
            except Exception:
                pass
        if _small_srcs:
            st.warning(
                f"⚠️ **원본이 {market_size}px보다 작은 파일이 {len(_small_srcs)}개 있습니다.** "
                "결과물은 규격에 맞게 저장되지만, 모자란 만큼 확대되어 화질이 뭉개집니다. "
                "가능하면 더 큰 원본으로 올려주세요 (스크린샷보다 원본 사진이 낫습니다).\n\n"
                + "\n".join(f"- {s}" for s in _small_srcs[:10]),
                icon="⚠️",
            )

    with st.expander("🔍 첫 번째 이미지로 미리보기 (슬라이더 조정 시 자동 갱신)", expanded=True):
        try:
            with st.spinner("미리보기 생성 중..."):
                # 미리보기도 저장될 때와 같은 정사각형 프레임으로 보여줍니다
                # (보이는 것과 저장되는 것이 다르면 미리보기의 의미가 없습니다).
                def _preview(im):
                    if market_size <= 0:
                        return im
                    return fit_to_square(im, 420)[0]
                # 같은 파일 객체를 미리보기·본처리에서 각각 열기 때문에
                # 읽기 위치를 매번 처음으로 되돌립니다(소싱도구에서 주소로
                # 받아온 이미지는 재사용되는 하나의 스트림입니다).
                try:
                    uploaded_files[0].seek(0)
                except Exception:
                    pass
                preview_src_full = normalize_to_rgb(Image.open(uploaded_files[0]))
                # 미리보기는 원본 해상도로 계산할 필요가 없음 - 큰 사진(특히 폰 카메라 원본)을 그대로
                # 쓰면 업로드/슬라이더 조작마다 몇 초씩 걸려 화면이 멈춘 것처럼 보이는 원인이 됨.
                # 화면에 보여주는 용도이므로 축소본으로 계산해서 항상 빠르게 반응하도록 함.
                preview_src = preview_src_full.copy()
                preview_src.thumbnail((700, 700), Image.LANCZOS)
                preview_tight = tighten_to_content(preview_src)
                preview_count = parse_count(stack_counts[0]) if stack_counts else parse_count(default_stack_count)

                row1 = st.columns(3)
                if opt_reflection:
                    refl_src = arrange_side_by_side(preview_tight, reflect_count, reflect_gap / 100) if reflect_count > 1 else preview_src
                    row1[0].markdown("**1. 미러 반사효과**")
                    row1[0].image(_preview(apply_mirror_reflection(refl_src, reflect_ratio, reflect_opacity, reflect_falloff)))
                if opt_stack:
                    row1[1].markdown("**2. 포개기 효과**")
                    row1[1].image(_preview(
                        apply_stack_fan(preview_tight, preview_count, stack_cols, stack_overlap, stack_angle, rise_ratio=stack_rise / 100, scale_step=stack_depth_scale / 100)
                    ))
                if opt_pastel:
                    row1[2].markdown("**3. 파스텔 배경**")
                    row1[2].image(_preview(
                        apply_pastel_bg(preview_tight, pastel_size, pastel_size, pastel_angle, lighten_amount=pastel_lighten / 100, shadow_shape=pastel_shadow_shape)
                    ))

                row2 = st.columns(3)
                if opt_dark:
                    row2[0].markdown("**4. 스튜디오톤 배경**")
                    row2[0].image(_preview(
                        apply_studio_dark_bg(preview_tight, dark_count, dark_gap / 100, reflect_opacity=dark_reflect_opacity,
                                              wall_bright=dark_wall_bright, floor_bright=dark_floor_bright,
                                              horizon_ratio=dark_horizon / 100, grain=dark_grain)
                    ))
                if opt_perspective:
                    row2[1].markdown("**5. 원근감 포개기**")
                    row2[1].image(_preview(
                        apply_perspective_fan(preview_tight, persp_count, persp_rotate, persp_scale / 100, persp_dx / 100, persp_dy / 100)
                    ))
        except Exception as e:
            st.warning(f"미리보기 생성 실패: {e}")

    if st.button("🚀 이미지 변환 및 자동 저장 시작", type="primary"):
        zip_buffer = io.BytesIO()
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(uploaded_files)
        upscale_warnings = []  # 원본이 작아 확대된 결과물 목록 (아래에서 안내)

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                base_name, ext = os.path.splitext(file_name)

                status_text.text(f"처리 중 ({idx + 1}/{total_files}): {file_name}")
                progress_bar.progress((idx + 1) / total_files)

                def save_to_zip(im, suffix):
                    # 모든 효과가 이 함수를 거쳐 저장되므로, 규격 맞추기는
                    # 여기 한 곳에서만 합니다 (효과를 추가해도 자동 적용).
                    squared, upscaled = fit_to_square(im, market_size)
                    if upscaled:
                        upscale_warnings.append(f"{file_name} → {suffix}")
                    buf = io.BytesIO()
                    squared.save(buf, format="JPEG", quality=95)
                    zip_file.writestr(f"{base_name}_{suffix}.jpg", buf.getvalue())

                try:
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass
                    orig_img = Image.open(uploaded_file)
                    original = normalize_to_rgb(orig_img)
                    tight = tighten_to_content(original)

                    if opt_reflection:
                        refl_src = arrange_side_by_side(tight, reflect_count, reflect_gap / 100) if reflect_count > 1 else original
                        save_to_zip(apply_mirror_reflection(refl_src, reflect_ratio, reflect_opacity, reflect_falloff), f"mirror_reflection_{reflect_count}pcs")

                    if opt_stack:
                        count = parse_count(stack_counts[idx]) if idx < len(stack_counts) else parse_count(default_stack_count)
                        save_to_zip(
                            apply_stack_fan(tight, count, stack_cols, stack_overlap, stack_angle, rise_ratio=stack_rise / 100, scale_step=stack_depth_scale / 100),
                            f"stacked_{count}pcs"
                        )

                    if opt_pastel:
                        save_to_zip(
                            apply_pastel_bg(tight, pastel_size, pastel_size, pastel_angle, lighten_amount=pastel_lighten / 100, shadow_shape=pastel_shadow_shape),
                            "pastel_bg"
                        )

                    if opt_dark:
                        save_to_zip(
                            apply_studio_dark_bg(tight, dark_count, dark_gap / 100, reflect_opacity=dark_reflect_opacity,
                                                  wall_bright=dark_wall_bright, floor_bright=dark_floor_bright,
                                                  horizon_ratio=dark_horizon / 100, grain=dark_grain),
                            f"dark_studio_{dark_count}pcs"
                        )

                    if opt_perspective:
                        save_to_zip(
                            apply_perspective_fan(tight, persp_count, persp_rotate, persp_scale / 100, persp_dx / 100, persp_dy / 100),
                            f"perspective_fan_{persp_count}pcs"
                        )

                    if opt_tight_double:
                        w, h = original.size
                        tight_double = Image.new("RGB", (w * 2, h), (255, 255, 255))
                        tight_double.paste(original, (0, 0))
                        tight_double.paste(original, (w, 0))
                        save_to_zip(tight_double, "super_tight_double")

                except Exception as e:
                    st.error(f"파일 처리 실패 ({file_name}): {e}")

        progress_bar.empty()
        status_text.empty()

        # 확대가 일어난 결과물 안내 — 코드로는 해결할 수 없고 더 큰 원본으로
        # 다시 만드는 수밖에 없어서, 그냥 넘어가지 않고 분명히 알려줍니다.
        if upscale_warnings:
            st.warning(
                f"⚠️ **원본이 작아 {len(upscale_warnings)}개 결과물이 확대되었습니다.** "
                f"규격({market_size}×{market_size})은 맞췄으니 마켓에 등록은 되지만, "
                "없는 픽셀을 늘린 것이라 화질이 다소 뭉개집니다. 더 선명하게 하시려면 "
                "**더 큰 원본 사진**으로 다시 만들어주세요 "
                f"(가로·세로 모두 {market_size}px 이상 권장, 휴대폰 카메라 원본이면 충분합니다).\n\n"
                + "\n".join(f"- {w}" for w in upscale_warnings[:10])
                + (f"\n- … 외 {len(upscale_warnings) - 10}개" if len(upscale_warnings) > 10 else ""),
                icon="⚠️",
            )

        zip_buffer.seek(0)

        # 로컬 PC(Windows)에서 직접 실행 중일 때만 지정 경로에 자동 저장 시도.
        # 클라우드(Streamlit Cloud 등)에 배포된 경우 서버 안에만 저장되고 사용자는 접근할 수 없으므로,
        # 그 경우엔 자동 저장을 건너뛰고 아래 다운로드 버튼만 사용하도록 함.
        if os.name == "nt":
            target_dir = r"C:\Users\PC\Pictures"
            try:
                os.makedirs(target_dir, exist_ok=True)
                save_path = os.path.join(target_dir, "processed_product_images.zip")
                with open(save_path, "wb") as f:
                    f.write(zip_buffer.getvalue())
                st.success(f"✨ 변환 완료! **{save_path}** 경로에 압축 파일이 성공적으로 저장되었습니다.")
            except Exception as e:
                st.warning(f"자동 저장 중 오류가 발생했습니다: {e}")
        else:
            st.success("✨ 변환 완료! 아래 버튼으로 다운로드하세요.")

        st.download_button(
            label="📦 변환된 전체 이미지 다운로드 (ZIP 수동 받기)",
            data=zip_buffer,
            file_name="processed_product_images.zip",
            mime="application/zip"
        )