"""Unit tests for the heads-up display and the title and end cards."""

import dataclasses
import unittest

import numpy as np

from generalist_robotics.viz import overlay

COPY = overlay.HudCopy(
    eyebrow="CONTINUATION IN MORPHOLOGY SPACE",
    title="Berkeley Humanoid · one policy, a body twice the size",
    steps_label="FINE-TUNE STEPS SPENT",
    steps_note="against 160,563,200 from scratch",
    footer="real time, not sped up · deterministic policy, seed 0",
    start_label="A · 1× SIZE",
    end_label="B · 2× SIZE",
)

MARKS = (
    overlay.TrackMark(0.0, "0.000"),
    overlay.TrackMark(0.475, "0.475", finetuned=True),
    overlay.TrackMark(1.0, "1.000"),
)

STATE = overlay.HudFrame(
    alpha=0.475,
    morphology=(
        overlay.Readout("SIZE", "× 1.39"),
        overlay.Readout("MASS", "× 2.69"),
        overlay.Readout("TORQUE", "× 3.73"),
    ),
    gait=(
        overlay.Readout("HIP HEIGHT", " 0.72 m"),
        overlay.Readout("SPEED", " 0.51 m/s"),
        overlay.Readout("FROUDE", " 0.037"),
    ),
    cumulative_steps=6_553_600,
    caption="The body is rescaled at every control step.",
)


def grey_frame(width: int, height: int, value: int = 200) -> np.ndarray:
    """A flat frame to compose the display onto."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def with_banner(state: overlay.HudFrame, banner: overlay.Banner) -> overlay.HudFrame:
    """Copy a HUD frame with a banner attached."""
    return dataclasses.replace(state, banner=banner)


class FontTest(unittest.TestCase):
    """Resolving the display faces."""

    def test_loads_every_declared_family(self):
        for family in overlay.FONT_FILES:
            self.assertIsNotNone(overlay.load_font(family, 20))

    def test_rejects_an_unknown_family(self):
        with self.assertRaises(KeyError):
            overlay.load_font("comic", 20)

    def test_the_same_request_is_cached(self):
        self.assertIs(overlay.load_font("mono", 18), overlay.load_font("mono", 18))


class LayoutTest(unittest.TestCase):
    """The display is laid out in design pixels and scaled to the frame."""

    def hud(self, width=1920, height=1080):
        """A display at one frame size."""
        return overlay.Hud(width, height, COPY, MARKS)

    def test_every_panel_is_inside_the_frame(self):
        for width, height in ((1920, 1080), (960, 540), (1280, 720)):
            hud = self.hud(width, height)
            boxes = (hud.title_box, hud.steps_box, hud.left_box, hud.right_box, hud.track_box)
            for box in boxes:
                self.assertGreaterEqual(box[0], 0)
                self.assertGreaterEqual(box[1], 0)
                self.assertLessEqual(box[2], width)
                self.assertLessEqual(box[3], height)

    def test_the_bottom_panels_do_not_overlap(self):
        hud = self.hud()
        self.assertLess(hud.left_box[2], hud.track_box[0])
        self.assertLess(hud.track_box[2], hud.right_box[0])

    def test_the_track_ends_sit_inside_its_panel(self):
        hud = self.hud()
        self.assertGreater(hud.track_left, hud.track_box[0])
        self.assertLess(hud.track_right, hud.track_box[2])

    def test_the_progress_marker_is_monotone_and_clamped(self):
        hud = self.hud()
        self.assertEqual(hud.track_x(0.0), hud.track_left)
        self.assertEqual(hud.track_x(1.0), hud.track_right)
        self.assertEqual(hud.track_x(-3.0), hud.track_left)
        self.assertEqual(hud.track_x(7.0), hud.track_right)
        self.assertLess(hud.track_x(0.25), hud.track_x(0.75))

    def test_layout_scales_with_the_frame(self):
        big = self.hud(1920, 1080)
        small = self.hud(960, 540)
        self.assertAlmostEqual(big.left_box[3] / small.left_box[3], 2.0, places=1)


class ComposeTest(unittest.TestCase):
    """Drawing the display onto a frame."""

    def hud(self, width=640, height=360):
        """A small display for fast composition."""
        return overlay.Hud(width, height, COPY, MARKS)

    def test_composition_keeps_the_frame_shape_and_type(self):
        frame = grey_frame(640, 360)
        composed = self.hud().draw(frame, STATE)
        self.assertEqual(composed.shape, frame.shape)
        self.assertEqual(composed.dtype, np.uint8)

    def test_composition_actually_marks_the_frame(self):
        frame = grey_frame(640, 360)
        composed = self.hud().draw(frame, STATE)
        self.assertGreater(np.abs(composed.astype(int) - frame.astype(int)).mean(), 1.0)

    def test_the_render_is_left_visible_in_the_middle_of_the_frame(self):
        frame = grey_frame(640, 360)
        composed = self.hud().draw(frame, STATE)
        middle = composed[150:200, 260:380]
        np.testing.assert_array_equal(middle, frame[150:200, 260:380])

    def test_the_banner_changes_the_frame_further(self):
        frame = grey_frame(640, 360)
        hud = self.hud()
        plain = hud.draw(frame, STATE)
        banner = overlay.Banner("FINE-TUNING", ("one line", "another"), progress=0.4, alert=True)
        called = hud.draw(frame, with_banner(STATE, banner))
        self.assertGreater(np.abs(called.astype(int) - plain.astype(int)).mean(), 1.0)

    def test_a_progress_bar_outside_zero_to_one_is_clamped(self):
        frame = grey_frame(640, 360)
        hud = self.hud()
        full = hud.draw(frame, with_banner(STATE, overlay.Banner("X", (), progress=1.0)))
        over = hud.draw(frame, with_banner(STATE, overlay.Banner("X", (), progress=9.0)))
        np.testing.assert_array_equal(full, over)

    def test_step_counts_are_grouped(self):
        self.assertEqual(overlay.format_steps(6553600), "6,553,600")
        self.assertEqual(overlay.format_steps(0), "0")


class CardTest(unittest.TestCase):
    """The title and end cards."""

    def test_card_has_the_requested_shape_and_is_not_blank(self):
        card = overlay.draw_card(
            640, 360, "RESULT", ("6,553,600 steps",), ("a line", ""), "footnote"
        )
        self.assertEqual(card.shape, (360, 640, 3))
        self.assertEqual(card.dtype, np.uint8)
        self.assertGreater(card.std(), 1.0)

    def test_card_background_is_the_palette_card_colour(self):
        card = overlay.draw_card(320, 180, "A", ("B",), ("C",), "D")
        np.testing.assert_array_equal(card[2, 2], np.array(overlay.Palette().card, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
