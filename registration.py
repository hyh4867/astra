"""Residual x/y translation estimate and correction for an A/B image pair.

Scope: the two images already share a pixel grid (same size; GeoTIFF pairs are
first reprojected by ``geospatial.align_pair``) and may still be offset by a
few pixels. Only a single global translation is handled; rotation, scale and
terrain parallax are not.

Convention used everywhere in this module:
    (dx, dy) = displacement of B's content relative to A, in pixels,
    +x to the right (columns), +y downward (rows).
    B(x, y) ~= A(x - dx, y - dy), so the correction moves B by (-dx, -dy).

The estimator reads the two images only. It never receives a label mask or an
injected shift.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil

import cv2
import numpy as np

STATUS_ALIGNED = "aligned"        # confident estimate below the noise floor: nothing to do
STATUS_SMALL = "small_shift"      # confident estimate below the auto-correction threshold: shown, not applied
STATUS_CORRECTED = "corrected"    # confident estimate: B was moved onto A's grid
STATUS_REVIEW = "review"          # ambiguous estimate: nothing applied, a person should look

REASON_TEXT = {
    "too_small": "영상이 너무 작아 이동을 추정할 수 없습니다.",
    "flat": "질감이 있는 구역이 부족합니다(평탄한 장면).",
    "repeated": "질감은 있지만 이동을 하나로 정할 수 없습니다(반복 패턴이거나 두 시점의 겉모습이 크게 다름).",
    "weak_peak": "영상 전체의 상관 피크가 뚜렷하지 않습니다.",
    "tiles_disagree": "구역마다 추정 이동이 다릅니다. 단순 평행 이동이 아닐 수 있습니다.",
    "out_of_range": "추정 이동이 자동 보정 범위를 넘습니다.",
}


@dataclass(frozen=True)
class RegistrationConfig:
    """Fixed settings. Values were chosen on synthetic data and the LEVIR-CD
    validation split only (scripts/select_registration_config.py)."""
    preprocess: str = "gray"          # "gray" or "gradient" (Sobel magnitude)
    estimator: str = "tile_median"    # "global" or "tile_median"
    search_px: int = 16               # correlation peak is searched within +-search_px
    max_auto_px: float = 12.0         # larger estimates are never applied automatically
    noise_floor_px: float = 0.5       # smaller estimates are reported as aligned
    min_shift_px: float = 4.0         # auto-correction threshold: smaller estimates leave B untouched
    refinement: str = "centroid"
    integer_correction: bool = True   # round the applied shift to whole pixels (no resampling blur)
    tile_px: int = 256
    tile_tol_px: float = 1.0          # a tile agrees when within this distance of the estimate
    min_tile_peak_ratio: float = 1.5  # a tile is informative above this peak / second-peak ratio
    flat_texture_std: float = 6.0     # grey-level std below which an uninformative tile counts as flat
    min_informative_fraction: float = 0.125
    min_tile_agreement: float = 0.5
    min_global_peak_ratio: float = 1.5


DEFAULT_CONFIG = RegistrationConfig()


@dataclass
class TileEstimate:
    row0: int
    col0: int
    size: int
    dx: float
    dy: float
    peak_ratio: float
    texture: float
    informative: bool
    agrees: bool


@dataclass
class RegistrationResult:
    status: str
    dx: float                 # estimated displacement of B relative to A
    dy: float
    applied_dx: float         # correction actually applied to B (0 when not applied)
    applied_dy: float
    applied: bool
    reasons: list[str]        # keys of REASON_TEXT; empty unless status == review
    after_aligned: np.ndarray # B on A's grid (identical to the input when not applied)
    valid_mask: np.ndarray    # True where after_aligned holds real B pixels
    global_dx: float
    global_dy: float
    global_peak_ratio: float
    informative_fraction: float
    tile_agreement: float
    tiles: list[TileEstimate] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.status == STATUS_REVIEW

    @property
    def reason_messages(self) -> list[str]:
        return [REASON_TEXT[key] for key in self.reasons]


def shift_image(image: np.ndarray, dx: float, dy: float, border=cv2.BORDER_REFLECT_101) -> np.ndarray:
    """Move image content by (dx, dy). Same call as the existing robustness
    experiment (scripts/scientific_evaluation.perturb) when dx == dy."""
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=border)


def shift_valid_mask(shape: tuple[int, int], dx: float, dy: float) -> np.ndarray:
    """True where ``shift_image(image, dx, dy)`` holds real (not padded) pixels."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x0 = max(0, ceil(dx)); x1 = min(w, w + int(np.floor(dx)))
    y0 = max(0, ceil(dy)); y1 = min(h, h + int(np.floor(dy)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _prepare(image: np.ndarray, preprocess: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    if preprocess == "gray":
        return gray
    if preprocess == "gradient":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)
    raise ValueError(f"Unknown preprocess: {preprocess}")


def _phase_peak(a: np.ndarray, b: np.ndarray, search_px: int, refinement: str = "centroid"):
    """Phase correlation of two equal-size float images.

    Returns (dx, dy, peak_ratio). peak_ratio compares the main peak with the
    highest value at least 3 px away inside the search window; a ratio near 1
    means another shift explains the pair almost as well."""
    h, w = a.shape
    radius = int(min(search_px, h // 2 - 1, w // 2 - 1))
    if radius < 2:
        return 0.0, 0.0, 0.0
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa = np.fft.rfft2((a - a.mean()) * window)
    fb = np.fft.rfft2((b - b.mean()) * window)
    cross = np.conj(fa) * fb
    magnitude = np.abs(cross)
    if not np.isfinite(magnitude).all() or magnitude.max() <= 1e-6:
        return 0.0, 0.0, 0.0
    cross /= magnitude + 1e-6 * magnitude.max()
    surface = np.fft.fftshift(np.fft.irfft2(cross, s=(h, w)))
    cy, cx = h // 2, w // 2
    region = surface[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
    py, px = np.unravel_index(int(np.argmax(region)), region.shape)
    peak = float(region[py, px])
    if peak <= 0:
        return 0.0, 0.0, 0.0
    yy, xx = np.ogrid[:region.shape[0], :region.shape[1]]
    away = (yy - py) ** 2 + (xx - px) ** 2 > 9
    second = float(region[away].max()) if away.any() else 0.0
    ratio = peak / max(second, 1e-9)
    # Sub-pixel position: centroid of the non-negative values in a 5x5 patch.
    y0, y1 = max(0, py - 2), min(region.shape[0], py + 3)
    x0, x1 = max(0, px - 2), min(region.shape[1], px + 3)
    patch = np.clip(region[y0:y1, x0:x1], 0, None)
    total = float(patch.sum())
    sub_y = float((patch.sum(axis=1) * np.arange(y0, y1)).sum() / total)
    sub_x = float((patch.sum(axis=0) * np.arange(x0, x1)).sum() / total)
    dx, dy = sub_x - radius, sub_y - radius
    if refinement == "dft":
        # Evaluate the inverse Fourier series only in a 2 px window around
        # the integer maximum. The conjugate half of the real FFT doubles
        # interior columns. DC and even-width Nyquist are present only once.
        offsets = np.arange(-10, 11) / 10
        xs, ys = px - radius + offsets, py - radius + offsets
        xphase = np.exp(2j * np.pi * np.outer(np.fft.rfftfreq(w), xs))
        xphase[1:(-1 if w % 2 == 0 else None)] *= 2
        yphase = np.exp(2j * np.pi * np.outer(ys, np.fft.fftfreq(h)))
        local = (yphase @ cross @ xphase).real
        iy, ix = np.unravel_index(local.argmax(), local.shape)
        dx, dy = xs[ix], ys[iy]
    elif refinement == "ecc":
        init = np.float32([[1, 0, dx], [0, 1, dy]])
        try:
            score, warp = cv2.findTransformECC(a, b, init, cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-5))
            dx, dy = warp[:, 2]
            if not np.isfinite([dx, dy, score]).all() or max(abs(dx),abs(dy)) > radius:
                return 0.0, 0.0, 0.0
        except cv2.error:
            return 0.0, 0.0, 0.0
    elif refinement != "centroid":
        raise ValueError(f"Unknown peak refinement: {refinement}")
    return float(dx), float(dy), float(min(ratio, 1e6))


def _tile_size(h: int, w: int, tile_px: int) -> int:
    return int(min(tile_px, max(64, min(h, w) // 2)))


def measure_shift(before: np.ndarray, after: np.ndarray, config: RegistrationConfig = DEFAULT_CONFIG):
    """Raw correlation measurements (whole image and tiles) from the two images
    alone. No thresholds are applied here, so the same measurement can be
    re-judged with different settings during validation."""
    if before.shape != after.shape:
        raise ValueError("Input images must have identical dimensions")
    h, w = before.shape[:2]
    if min(h, w) < 32:
        return {"too_small": True, "global": (0.0, 0.0, 0.0), "tile_size": 0, "tiles": []}
    gray_a = _prepare(before, "gray"); gray_b = _prepare(after, "gray")
    a = _prepare(before, config.preprocess)
    b = _prepare(after, config.preprocess)
    size = _tile_size(h, w, config.tile_px)
    tiles = []
    if min(h, w) >= 64:
        rows = max(1, h // size); cols = max(1, w // size)
        # Spread the tiles evenly so the right/bottom remainder is not ignored.
        for r0 in np.linspace(0, h - size, rows).round().astype(int):
            for c0 in np.linspace(0, w - size, cols).round().astype(int):
                tdx, tdy, tratio = _phase_peak(a[r0:r0 + size, c0:c0 + size], b[r0:r0 + size, c0:c0 + size], config.search_px, config.refinement)
                texture = float(min(gray_a[r0:r0 + size, c0:c0 + size].std(), gray_b[r0:r0 + size, c0:c0 + size].std()))
                tiles.append((int(r0), int(c0), tdx, tdy, tratio, texture))
    return {"too_small": False, "global": _phase_peak(a, b, config.search_px, config.refinement), "tile_size": size, "tiles": tiles}


def judge_measurement(measurement: dict, config: RegistrationConfig = DEFAULT_CONFIG):
    """Turn a raw measurement into an estimate plus ambiguity reasons
    (empty list = the estimate is considered reliable)."""
    if measurement["too_small"]:
        return {"dx": 0.0, "dy": 0.0, "global_dx": 0.0, "global_dy": 0.0, "global_peak_ratio": 0.0,
                "informative_fraction": 0.0, "tile_agreement": 0.0, "tiles": [], "reasons": ["too_small"]}
    gdx, gdy, gratio = measurement["global"]
    raw = measurement["tiles"]; size = measurement["tile_size"]
    informative = [t for t in raw if t[4] >= config.min_tile_peak_ratio]
    informative_fraction = len(informative) / len(raw) if raw else 0.0

    if config.estimator == "tile_median" and informative:
        dx = float(np.median([t[2] for t in informative]))
        dy = float(np.median([t[3] for t in informative]))
    else:
        dx, dy = gdx, gdy

    def near(t):
        return float(np.hypot(t[2] - dx, t[3] - dy)) <= config.tile_tol_px

    agreement = sum(near(t) for t in informative) / len(informative) if informative else 0.0
    tiles = [TileEstimate(t[0], t[1], size, t[2], t[3], t[4], t[5], t[4] >= config.min_tile_peak_ratio,
                          t[4] >= config.min_tile_peak_ratio and near(t)) for t in raw]

    reasons = []
    few = bool(raw) and informative_fraction < config.min_informative_fraction
    if few:
        # Too few usable tiles: flat when the unusable tiles lack contrast,
        # otherwise textured-but-ambiguous (repeated pattern or changed appearance).
        unusable = [t for t in raw if t[4] < config.min_tile_peak_ratio]
        low_contrast = sum(t[5] < config.flat_texture_std for t in unusable)
        reasons.append("flat" if low_contrast * 2 >= len(unusable) else "repeated")
    elif raw and agreement < config.min_tile_agreement:
        reasons.append("tiles_disagree")
    if gratio < config.min_global_peak_ratio:
        reasons.append("weak_peak")
    if float(np.hypot(dx, dy)) > config.max_auto_px:
        reasons.append("out_of_range")
    return {"dx": dx, "dy": dy, "global_dx": gdx, "global_dy": gdy, "global_peak_ratio": gratio,
            "informative_fraction": informative_fraction, "tile_agreement": agreement,
            "tiles": tiles, "reasons": reasons}


def estimate_shift(before: np.ndarray, after: np.ndarray, config: RegistrationConfig = DEFAULT_CONFIG):
    """Estimate B's displacement relative to A from the two images alone."""
    return judge_measurement(measure_shift(before, after, config), config)


def register_pair(before: np.ndarray, after: np.ndarray, config: RegistrationConfig = DEFAULT_CONFIG) -> RegistrationResult:
    """Estimate the residual shift and, only when the estimate is reliable,
    move B onto A's grid. Ambiguous pairs are returned untouched with
    status ``review``."""
    est = estimate_shift(before, after, config)
    h, w = before.shape[:2]
    full = np.ones((h, w), dtype=bool)
    common = dict(dx=est["dx"], dy=est["dy"], global_dx=est["global_dx"], global_dy=est["global_dy"],
                  global_peak_ratio=est["global_peak_ratio"], informative_fraction=est["informative_fraction"],
                  tile_agreement=est["tile_agreement"], tiles=est["tiles"])
    if est["reasons"]:
        return RegistrationResult(status=STATUS_REVIEW, applied_dx=0.0, applied_dy=0.0, applied=False,
                                  reasons=est["reasons"], after_aligned=after, valid_mask=full, **common)
    move_x, move_y = -est["dx"], -est["dy"]
    if config.integer_correction:
        move_x, move_y = float(round(move_x)), float(round(move_y))
    magnitude = float(np.hypot(est["dx"], est["dy"]))
    if magnitude < config.min_shift_px or (move_x == 0 and move_y == 0):
        status = STATUS_ALIGNED if magnitude < config.noise_floor_px else STATUS_SMALL
        return RegistrationResult(status=status, applied_dx=0.0, applied_dy=0.0, applied=False,
                                  reasons=[], after_aligned=after, valid_mask=full, **common)
    return RegistrationResult(status=STATUS_CORRECTED, applied_dx=move_x, applied_dy=move_y, applied=True,
                              reasons=[], after_aligned=shift_image(after, move_x, move_y),
                              valid_mask=shift_valid_mask((h, w), move_x, move_y), **common)


def with_overrides(config: RegistrationConfig = DEFAULT_CONFIG, **changes) -> RegistrationConfig:
    return replace(config, **changes)


def tile_diagnostic(image: np.ndarray, result: RegistrationResult) -> np.ndarray:
    """Draw the tile grid on an RGB image: green = agrees with the estimate,
    red = informative but disagrees (inspect here), grey = too flat to use."""
    out = image.copy()
    thickness = max(2, min(image.shape[:2]) // 300)
    for tile in result.tiles:
        if tile.agrees:
            color = (40, 200, 90)
        elif tile.informative:
            color = (235, 50, 50)
        else:
            color = (150, 150, 150)
        inset = thickness
        cv2.rectangle(out, (tile.col0 + inset, tile.row0 + inset),
                      (tile.col0 + tile.size - inset - 1, tile.row0 + tile.size - inset - 1), color, thickness)
        if tile.informative and not tile.agrees:
            overlay = out[tile.row0:tile.row0 + tile.size, tile.col0:tile.col0 + tile.size]
            overlay[:] = (0.75 * overlay + 0.25 * np.array(color)).astype(np.uint8)
    return out


def alignment_composite(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Red = B, cyan = A. Aligned structure looks grey; an offset shows up as
    red/cyan fringes along edges."""
    ga = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    return np.dstack([gb, ga, ga])


def zoom_window(shape: tuple[int, int], tile: TileEstimate | None, size: int = 160):
    """Square crop (row0, row1, col0, col1) centred on a tile, or on the image."""
    h, w = shape[:2]
    size = int(min(size, h, w))
    cy, cx = (tile.row0 + tile.size // 2, tile.col0 + tile.size // 2) if tile else (h // 2, w // 2)
    r0 = int(np.clip(cy - size // 2, 0, h - size)); c0 = int(np.clip(cx - size // 2, 0, w - size))
    return r0, r0 + size, c0, c0 + size


def magnify(image: np.ndarray, factor: int = 3) -> np.ndarray:
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
