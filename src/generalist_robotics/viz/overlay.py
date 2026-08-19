"""Composition of the demo's heads-up display and its title and end cards."""

import dataclasses
import functools
import importlib.util
import pathlib
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Fonts are resolved by trying directories in order, so the demo renders the same on a machine
# that ships DejaVu system-wide and on one that only has matplotlib's bundled copy. Numbers are
# set in a monospaced face: a proportional one makes a live readout jitter as its digits change.
SYSTEM_FONT_DIRECTORIES = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
)
FONT_FILES = {
    "sans": "DejaVuSans.ttf",
    "sans_bold": "DejaVuSans-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
    "mono_bold": "DejaVuSansMono-Bold.ttf",
}

# The display is laid out once, in the pixels of a 1080p frame, and scaled to whatever the
# render actually is. Mixing absolute offsets with relative ones is what makes a layout that
# looks composed at one resolution collide with itself at another.
DESIGN_HEIGHT = 1080


@dataclasses.dataclass(frozen=True)
class Palette:
    """Ink and panel colours, chosen to sit on a bright render without glowing.

    The scheme is a printed page laid over the shot: near-black text on off-white panels, one
    warm accent reserved for the fine-tuning beat and one cool accent for progress.
    """

    ink: tuple[int, int, int] = (22, 24, 28)
    ink_soft: tuple[int, int, int] = (74, 80, 88)
    muted: tuple[int, int, int] = (122, 130, 140)
    panel: tuple[int, int, int, int] = (252, 252, 250, 222)
    panel_edge: tuple[int, int, int, int] = (204, 208, 202, 255)
    rule: tuple[int, int, int] = (201, 205, 198)
    warm: tuple[int, int, int] = (183, 58, 34)
    cool: tuple[int, int, int] = (15, 110, 107)
    card: tuple[int, int, int] = (250, 249, 246)


@dataclasses.dataclass(frozen=True)
class Readout:
    """One labelled number in a HUD panel."""

    label: str
    value: str


@dataclasses.dataclass(frozen=True)
class TrackMark:
    """One waypoint tick on the progress track from the start body to the target body."""

    alpha: float
    label: str
    finetuned: bool = False


@dataclasses.dataclass(frozen=True)
class Banner:
    """The centred state callout, raised for the fine-tuning beat.

    Attributes:
        title: the state, set in caps.
        lines: explanatory lines under it.
        progress: 0..1 fill of the bar under the text, or None for no bar.
        alert: whether to draw it in the warm accent rather than the cool one.
    """

    title: str
    lines: tuple[str, ...]
    progress: float | None = None
    alert: bool = False


@dataclasses.dataclass(frozen=True)
class HudCopy:
    """The strings that never change during a clip."""

    eyebrow: str
    title: str
    steps_label: str
    steps_note: str
    footer: str
    start_label: str
    end_label: str


@dataclasses.dataclass(frozen=True)
class HudFrame:
    """Everything the HUD draws that changes from frame to frame."""

    alpha: float
    morphology: tuple[Readout, ...]
    gait: tuple[Readout, ...]
    cumulative_steps: int
    caption: str = ""
    banner: Banner | None = None


@functools.lru_cache(maxsize=1)
def font_directories() -> tuple[pathlib.Path, ...]:
    """Directories searched for the display faces, system copies first.

    matplotlib ships the same DejaVu faces inside its own package, which is the fallback for a
    machine whose system font set is bare; its location is read off the module spec so the
    package is not imported for it.
    """
    directories = [pathlib.Path(name) for name in SYSTEM_FONT_DIRECTORIES]
    spec = importlib.util.find_spec("matplotlib")
    if spec is not None and spec.origin is not None:
        directories.append(pathlib.Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf")
    return tuple(directories)


@functools.lru_cache(maxsize=64)
def load_font(family: str, size: int) -> ImageFont.FreeTypeFont:
    """Return a font of the given family and pixel size, from the first directory that has it.

    Raises:
        KeyError: if the family is not one of FONT_FILES.
        FileNotFoundError: if no candidate directory carries the family.
    """
    if family not in FONT_FILES:
        raise KeyError(f"unknown font family {family!r}; known: {sorted(FONT_FILES)}")
    for directory in font_directories():
        path = directory / FONT_FILES[family]
        if path.exists():
            return ImageFont.truetype(str(path), max(1, size))
    raise FileNotFoundError(f"no {FONT_FILES[family]} under any of {font_directories()}")


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    """Width in pixels of a string set in a font."""
    return draw.textlength(text, font=font)


def tracked_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: float
) -> float:
    """Width in pixels of a letterspaced string."""
    if not text:
        return 0.0
    return text_width(draw, text, font) + tracking * (len(text) - 1)


def draw_tracked(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    tracking: float = 2.0,
) -> None:
    """Draw letterspaced text, which is how the small caps labels are set."""
    x, y = position
    for character in text:
        draw.text((x, y), character, font=font, fill=fill)
        x += draw.textlength(character, font=font) + tracking


def draw_panel(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], palette: Palette, radius: int = 10
) -> None:
    """Draw one translucent panel with a hairline edge."""
    draw.rounded_rectangle(box, radius=radius, fill=palette.panel, outline=palette.panel_edge)


def format_steps(steps: int) -> str:
    """Group a step count with thousands separators."""
    return f"{steps:,}"


class Hud:
    """The heads-up display: fixed chrome drawn once, live numbers drawn per frame.

    The chrome — panels, labels, the progress track and its waypoint ticks — is composed into a
    single RGBA layer at construction and alpha-composited onto every frame, so the per-frame
    cost is only the strings that actually move. Every position is a constant in 1080p pixels
    scaled by the frame height, which keeps the layout identical at preview and delivery size
    and keeps it from twitching as digits change width.
    """

    def __init__(
        self,
        width: int,
        height: int,
        copy: HudCopy,
        marks: Sequence[TrackMark],
        palette: Palette | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.copy = copy
        self.marks = tuple(marks)
        self.palette = palette if palette is not None else Palette()
        self.scale = height / DESIGN_HEIGHT
        self.side = (width - height * 16 / 9) / 2  # Pillarbox if the frame is wider than 16:9.

        self.title_box = self.box(56, 46, 756, 158)
        self.caption_top = self.px(176)
        self.steps_box = self.box(1394, 46, 1864, 176)
        self.left_box = self.box(56, 834, 486, 1010)
        self.right_box = self.box(1434, 834, 1864, 1010)
        self.track_box = self.box(546, 834, 1374, 1010)
        self.track_left = self.x(602)
        self.track_right = self.x(1318)
        self.track_y = self.y(946)
        self.label_y = self.y(852)
        self.footer_y = self.y(1026)
        self.banner_box = self.box(330, 176, 1590, 366)
        self.chrome = self.build_chrome()

    def px(self, value: float) -> int:
        """Convert a length in 1080p design pixels to this frame's pixels."""
        return round(value * self.scale)

    def x(self, value: float) -> int:
        """Convert an x coordinate in a 1920x1080 design frame to this frame."""
        return round(self.side + value * self.scale)

    def y(self, value: float) -> int:
        """Convert a y coordinate in a 1920x1080 design frame to this frame."""
        return round(value * self.scale)

    def box(
        self, left: float, top: float, right: float, bottom: float
    ) -> tuple[int, int, int, int]:
        """Convert a design-space rectangle to this frame."""
        return (self.x(left), self.y(top), self.x(right), self.y(bottom))

    def font(self, family: str, size: float) -> ImageFont.FreeTypeFont:
        """Load a face at a design-space point size scaled to this frame."""
        return load_font(family, self.px(size))

    def track_x(self, alpha: float) -> float:
        """Pixel x of a path coordinate on the progress track."""
        span = self.track_right - self.track_left
        return self.track_left + span * min(1.0, max(0.0, alpha))

    def build_chrome(self) -> Image.Image:
        """Compose the parts of the display that never change into one RGBA layer."""
        palette = self.palette
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        eyebrow = self.font("sans_bold", 14)
        tick = self.font("sans", 14)
        heading = self.font("sans_bold", 14)
        title = self.font("sans_bold", 32)
        tracking = self.px(2)

        draw_panel(draw, self.title_box, palette)
        draw_tracked(
            draw, (self.x(78), self.y(66)), self.copy.eyebrow, eyebrow, palette.cool, tracking
        )
        draw.text((self.x(78), self.y(94)), self.copy.title, font=title, fill=palette.ink)

        draw_panel(draw, self.steps_box, palette)
        width = tracked_width(draw, self.copy.steps_label, eyebrow, tracking)
        draw_tracked(
            draw,
            (self.x(1842) - width, self.y(66)),
            self.copy.steps_label,
            eyebrow,
            palette.muted,
            tracking,
        )
        note = self.font("sans", 16)
        draw.text(
            (self.x(1842) - text_width(draw, self.copy.steps_note, note), self.y(146)),
            self.copy.steps_note,
            font=note,
            fill=palette.muted,
        )

        for box, name in ((self.left_box, "MORPHOLOGY"), (self.right_box, "GAIT")):
            draw_panel(draw, box, palette)
            draw_tracked(
                draw,
                (box[0] + self.px(22), box[1] + self.px(18)),
                name,
                heading,
                palette.muted,
                tracking,
            )

        draw_panel(draw, self.track_box, palette)
        draw_tracked(
            draw,
            (self.x(602), self.label_y),
            self.copy.start_label,
            heading,
            palette.ink_soft,
            tracking,
        )
        end_width = tracked_width(draw, self.copy.end_label, heading, tracking)
        draw_tracked(
            draw,
            (self.x(1318) - end_width, self.label_y),
            self.copy.end_label,
            heading,
            palette.ink_soft,
            tracking,
        )
        draw.rounded_rectangle(
            (
                self.track_left,
                self.track_y - self.px(3),
                self.track_right,
                self.track_y + self.px(3),
            ),
            radius=self.px(3),
            fill=palette.rule,
        )
        for mark in self.marks:
            x = self.track_x(mark.alpha)
            colour = palette.warm if mark.finetuned else palette.ink_soft
            reach = self.px(16 if mark.finetuned else 11)
            half = max(1, self.px(1.5))
            draw.rectangle((x - half, self.track_y - reach, x + half, self.track_y + reach), colour)
            if mark.label:
                anchor = x - text_width(draw, mark.label, tick) / 2
                draw.text(
                    (anchor, self.track_y + reach + self.px(6)),
                    mark.label,
                    font=tick,
                    fill=colour,
                )

        footer = self.font("sans", 15)
        draw.text(
            (self.x(56), self.footer_y + self.px(1)),
            self.copy.footer,
            font=footer,
            fill=(246, 246, 243),
        )
        draw.text((self.x(56), self.footer_y), self.copy.footer, font=footer, fill=palette.ink_soft)
        return layer

    def draw_readouts(
        self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], rows: Sequence[Readout]
    ) -> None:
        """Print a panel's rows: label on the left, monospaced value on the right."""
        label = self.font("sans", 17)
        value = self.font("mono_bold", 25)
        for index, row in enumerate(rows):
            y = box[1] + self.px(52 + 42 * index)
            draw.text(
                (box[0] + self.px(22), y + self.px(5)),
                row.label,
                font=label,
                fill=self.palette.muted,
            )
            width = text_width(draw, row.value, value)
            draw.text(
                (box[2] - self.px(22) - width, y),
                row.value,
                font=value,
                fill=self.palette.ink,
            )

    def draw_caption(self, image: Image.Image, caption: str) -> None:
        """Draw the beat's one-line caption on a panel sized to it, under the title."""
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = self.font("sans", 18)
        left = self.x(56)
        box = (
            left,
            self.caption_top,
            left + round(text_width(draw, caption, font)) + self.px(44),
            self.caption_top + self.px(46),
        )
        draw_panel(draw, box, self.palette, radius=self.px(8))
        draw.text(
            (left + self.px(22), box[1] + self.px(12)),
            caption,
            font=font,
            fill=self.palette.ink_soft,
        )
        image.alpha_composite(layer)

    def draw_banner(self, image: Image.Image, banner: Banner) -> None:
        """Draw the centred state callout over the empty sky above the robot."""
        palette = self.palette
        accent = palette.warm if banner.alert else palette.cool
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        box = self.banner_box
        draw.rounded_rectangle(
            box,
            radius=self.px(12),
            fill=(252, 252, 250, 238),
            outline=accent,
            width=max(1, self.px(3)),
        )
        title = self.font("sans_bold", 30)
        body = self.font("sans", 18)
        centre = (box[0] + box[2]) / 2
        tracking = self.px(4)

        width = tracked_width(draw, banner.title, title, tracking)
        draw_tracked(draw, (centre - width / 2, self.y(198)), banner.title, title, accent, tracking)
        for index, line in enumerate(banner.lines):
            y = self.y(246 + 28 * index)
            draw.text(
                (centre - text_width(draw, line, body) / 2, y),
                line,
                font=body,
                fill=palette.ink_soft,
            )
        if banner.progress is not None:
            bar = (self.x(364), self.y(326), self.x(1556), self.y(338))
            radius = self.px(6)
            draw.rounded_rectangle(bar, radius=radius, fill=palette.rule)
            filled = bar[0] + (bar[2] - bar[0]) * min(1.0, max(0.0, banner.progress))
            if filled > bar[0] + 2 * radius:
                draw.rounded_rectangle((bar[0], bar[1], filled, bar[3]), radius=radius, fill=accent)
        image.alpha_composite(layer)

    def draw(self, frame: np.ndarray, state: HudFrame) -> np.ndarray:
        """Compose the display onto one rendered frame and return the result."""
        image = Image.fromarray(frame).convert("RGBA")
        image.alpha_composite(self.chrome)
        draw = ImageDraw.Draw(image)

        steps_font = self.font("mono_bold", 46)
        steps = format_steps(state.cumulative_steps)
        colour = self.palette.ink if state.cumulative_steps else self.palette.muted
        draw.text(
            (self.x(1842) - text_width(draw, steps, steps_font), self.y(88)),
            steps,
            font=steps_font,
            fill=colour,
        )

        self.draw_readouts(draw, self.left_box, state.morphology)
        self.draw_readouts(draw, self.right_box, state.gait)

        alpha_font = self.font("mono_bold", 24)
        text = f"α = {state.alpha:0.3f}"
        centre = (self.track_left + self.track_right) / 2
        draw.text(
            (centre - text_width(draw, text, alpha_font) / 2, self.label_y - self.px(4)),
            text,
            font=alpha_font,
            fill=self.palette.ink,
        )

        x = self.track_x(state.alpha)
        draw.rounded_rectangle(
            (self.track_left, self.track_y - self.px(3), x, self.track_y + self.px(3)),
            radius=self.px(3),
            fill=self.palette.cool,
        )
        outer, inner = self.px(11), self.px(5)
        draw.ellipse(
            (x - outer, self.track_y - outer, x + outer, self.track_y + outer), self.palette.cool
        )
        draw.ellipse(
            (x - inner, self.track_y - inner, x + inner, self.track_y + inner), (252, 252, 250)
        )

        if state.caption:
            self.draw_caption(image, state.caption)
        if state.banner is not None:
            self.draw_banner(image, state.banner)
        return np.asarray(image.convert("RGB"))


def draw_card(
    width: int,
    height: int,
    eyebrow: str,
    headline: Sequence[str],
    body: Sequence[str],
    footer: str,
    palette: Palette | None = None,
) -> np.ndarray:
    """Compose a full-frame title or end card and return it as a uint8 RGB array."""
    palette = palette if palette is not None else Palette()
    image = Image.new("RGB", (width, height), palette.card)
    draw = ImageDraw.Draw(image)
    scale = height / DESIGN_HEIGHT
    left = round(width * 0.115)

    eyebrow_font = load_font("sans_bold", round(17 * scale))
    headline_font = load_font("sans_bold", round(66 * scale))
    body_font = load_font("sans", round(25 * scale))
    footer_font = load_font("sans", round(17 * scale))

    y = round(280 * scale)
    draw_tracked(draw, (left, y), eyebrow, eyebrow_font, palette.cool, round(5 * scale))
    y += round(58 * scale)
    for line in headline:
        draw.text((left, y), line, font=headline_font, fill=palette.ink)
        y += round(84 * scale)
    y += round(20 * scale)
    draw.rounded_rectangle(
        (left, y, left + round(96 * scale), y + round(5 * scale)), radius=2, fill=palette.warm
    )
    y += round(46 * scale)
    for line in body:
        draw.text((left, y), line, font=body_font, fill=palette.ink_soft)
        y += round(41 * scale)
    draw.text((left, height - round(96 * scale)), footer, font=footer_font, fill=palette.muted)
    return np.asarray(image)
