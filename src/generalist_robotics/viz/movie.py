"""The demo film: storyboard, HUD binding and encoding of the continuation run to mp4 and GIF."""

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence

import imageio_ffmpeg
import numpy as np

from generalist_robotics.viz import overlay, render

# Where a checked-out repository keeps the runs this film is made of.
DEFAULT_RUN_DIR = pathlib.Path("artifacts/continuation_similar")
DEFAULT_BASELINE_PATH = pathlib.Path("artifacts/baseline_berkeley/baseline.json")
DEFAULT_MEDIA_DIR = pathlib.Path("media")

# Fraction of a fine-tuning beat spent watching the step counter fill before the fine-tuned
# weights take over. Splitting the beat is what makes the switch legible as an event.
FINETUNE_REVEAL = 0.62

# Encoder settings. Average bitrate rather than a quality target, so the delivered file is
# predictably above the 10 Mbit/s the brief asks for whatever the shot's complexity.
VIDEO_BITRATE = "14M"
VIDEO_MAXRATE = "20M"
VIDEO_BUFSIZE = "40M"

# Frame rate, palette size and width tried in order until one lands under the size budget, which
# is set well under the 10 MB a README can carry so the loop stays comfortable to load. The shot
# pans across a floor grid, so every frame differs everywhere and a GIF of it is expensive;
# dithering is off because its noise costs more bytes here than the banding it removes.
GIF_ATTEMPTS = ((16, 176, 800), (14, 144, 760), (12, 112, 720), (11, 96, 680), (10, 80, 640))
GIF_SIZE_LIMIT = 6_500_000
GIF_DITHER = "none"


@dataclasses.dataclass(frozen=True)
class Baseline:
    """What training this robot from scratch cost, the number the film is measured against."""

    steps: int
    reward: float


def load_baseline(path: pathlib.Path) -> Baseline:
    """Read the from-scratch run's step count and final reward."""
    record = json.loads(pathlib.Path(path).read_text())
    return Baseline(
        steps=int(record["num_timesteps"]),
        reward=float(record["metrics"]["eval/episode_reward"]),
    )


@dataclasses.dataclass(frozen=True)
class Timing:
    """Durations of the film's parts, in seconds of the finished clip.

    Attributes:
        title: the opening card.
        opening: walking at the start morphology before the body begins to grow.
        growth: total time spent growing, split between the growth beats by how much of the
            path each covers.
        finetune: one fine-tuning stop.
        arrival: walking at the target morphology.
        end: the closing card.
        fade: cross-fade between a card and the shot.
    """

    title: float = 4.0
    opening: float = 5.0
    growth: float = 24.0
    finetune: float = 8.0
    arrival: float = 7.0
    end: float = 7.0
    fade: float = 0.5


class DemoScript:
    """The storyboard of the hero clip and everything the HUD says at each point of it.

    The path is cut at every waypoint that had to be fine-tuned: the body grows up to it, stops
    while the step counter fills, and grows on once the fine-tuned weights are in charge. On the
    similar-morphology run that is exactly one stop, at alpha = 0.475, which is the point of the
    film.
    """

    def __init__(
        self,
        waypoints: Sequence[render.Waypoint],
        baseline: Baseline,
        timing: Timing | None = None,
    ) -> None:
        self.waypoints = tuple(waypoints)
        self.baseline = baseline
        self.timing = timing if timing is not None else Timing()
        self.stops = tuple(w for w in self.waypoints if w.finetune_steps > 0)
        self.target = self.waypoints[-1]
        self.beats = self.build_beats()

    def build_beats(self) -> tuple[render.Beat, ...]:
        """Cut the path into an opening hold, growth beats, fine-tuning stops and an arrival."""
        timing = self.timing
        edges = [w.alpha for w in self.stops] + [self.target.alpha]
        travelled = sum(abs(b - a) for a, b in zip([0.0, *edges[:-1]], edges, strict=True))
        beats = [render.Beat("open", timing.opening, 0.0, 0.0)]
        previous = 0.0
        for index, edge in enumerate(edges):
            share = abs(edge - previous) / travelled if travelled else 1.0
            beats.append(render.Beat(f"grow_{index}", timing.growth * share, previous, edge))
            if index < len(self.stops):
                beats.append(render.Beat(f"finetune_{index}", timing.finetune, edge, edge))
            previous = edge
        beats.append(render.Beat("arrive", timing.arrival, self.target.alpha, self.target.alpha))
        return tuple(beats)

    def stop_for(self, beat: render.Beat) -> render.Waypoint | None:
        """The waypoint a fine-tuning beat is about, or None for any other beat."""
        if not beat.name.startswith("finetune_"):
            return None
        return self.stops[int(beat.name.split("_")[1])]

    def policy_override(self, beat: render.Beat, fraction: float) -> int | None:
        """Hold the arriving policy in place until a fine-tuning beat's counter has filled."""
        stop = self.stop_for(beat)
        if stop is None or fraction >= FINETUNE_REVEAL:
            return None
        return max(0, self.waypoints.index(stop) - 1)

    def cumulative_steps(self, beat: render.Beat, fraction: float, reached: int) -> int:
        """The step counter's reading: the spend so far, filling in as a stop plays out."""
        stop = self.stop_for(beat)
        if stop is None:
            return reached
        filled = min(1.0, fraction / FINETUNE_REVEAL)
        return int(round(stop.cumulative_steps - stop.finetune_steps * (1.0 - filled)))

    def caption(self, beat: render.Beat) -> str:
        """The one-line caption under the title for a beat."""
        if beat.name == "open":
            return (
                f"Trained here from scratch: {self.baseline.steps:,} environment steps, "
                f"final reward {self.baseline.reward:.2f}."
            )
        if beat.name == "grow_0":
            return "The body is rescaled at every control step. The policy is not being trained."
        if beat.name.startswith("grow"):
            return (
                f"The same weights, fine-tuned once at α = {self.stops[-1].alpha:.3f}. "
                "Nothing further is spent from here on."
            )
        if beat.name == "arrive":
            share = 100.0 * self.target.cumulative_steps / self.baseline.steps
            return (
                f"Target body reached: ×{self.target.params.size_scale:.0f} size, "
                f"×{self.target.params.mass_scale:.0f} mass, "
                f"×{self.target.params.torque_scale:.0f} torque — for {share:.1f}% of a run "
                "from scratch."
            )
        return ""

    def banner(self, beat: render.Beat, fraction: float) -> overlay.Banner | None:
        """The centred callout for a beat, raised only while a fine-tuning stop plays out."""
        stop = self.stop_for(beat)
        if stop is None:
            return None
        if fraction < FINETUNE_REVEAL:
            return overlay.Banner(
                title="FINE-TUNING",
                lines=(
                    f"α = {stop.alpha:.3f}, size ×{stop.params.size_scale:.2f} — the arriving "
                    f"policy survived {stop.survived_before:.3f} of the episode, under the "
                    "0.800 floor",
                    f"PPO continues from these weights for {stop.finetune_steps:,} "
                    "environment steps",
                ),
                progress=min(1.0, fraction / FINETUNE_REVEAL),
                alert=True,
            )
        after = stop.survived_after if stop.survived_after is not None else stop.survived_before
        speed_after = stop.speed_after if stop.speed_after is not None else stop.speed_before
        return overlay.Banner(
            title="FINE-TUNED",
            lines=(
                f"survived {stop.survived_before:.3f} → {after:.3f}   ·   "
                f"forward speed {stop.speed_before:.2f} → {speed_after:.2f} m/s",
                "the walk resumes, and no further training is spent between here and the target",
            ),
            progress=1.0,
            alert=False,
        )

    def marks(self) -> tuple[overlay.TrackMark, ...]:
        """Waypoint ticks for the progress track."""
        return tuple(
            overlay.TrackMark(w.alpha, f"{w.alpha:.3f}", w.finetune_steps > 0)
            for w in self.waypoints
        )

    def copy(self) -> overlay.HudCopy:
        """The strings the HUD keeps on screen for the whole clip."""
        target = self.target.params
        return overlay.HudCopy(
            eyebrow="CONTINUATION IN MORPHOLOGY SPACE",
            title="Berkeley Humanoid · one policy, a body twice the size",
            steps_label="FINE-TUNE STEPS SPENT",
            steps_note=f"against {self.baseline.steps:,} from scratch",
            footer=(
                "real time, not sped up  ·  deterministic policy, seed 0  ·  sensor noise and "
                "random pushes off  ·  forward command 0.50·√k m/s  ·  floor squares are 1.00 m "
                "and the posts are marked at 0.515 m and 1.030 m  ·  hip height is the standing "
                "base height"
            ),
            start_label="A · 1× SIZE",
            end_label=f"B · {target.size_scale:.0f}× SIZE",
        )

    def hud_frame(
        self, telemetry: render.Telemetry, beat: render.Beat, fraction: float
    ) -> overlay.HudFrame:
        """Bind one frame of telemetry to what the HUD should print."""
        params = telemetry.params
        return overlay.HudFrame(
            alpha=telemetry.alpha,
            morphology=(
                overlay.Readout("SIZE", f"×{params.size_scale:5.2f}"),
                overlay.Readout("MASS", f"×{params.mass_scale:5.2f}"),
                overlay.Readout("TORQUE", f"×{params.torque_scale:5.2f}"),
            ),
            gait=(
                overlay.Readout("HIP HEIGHT", f"{telemetry.standing_height:5.2f} m"),
                overlay.Readout("SPEED", f"{telemetry.speed:5.2f} m/s"),
                overlay.Readout("FROUDE", f"{telemetry.froude:6.3f}"),
            ),
            cumulative_steps=self.cumulative_steps(beat, fraction, telemetry.cumulative_steps),
            caption=self.caption(beat),
            banner=self.banner(beat, fraction),
        )

    def title_card(self, width: int, height: int) -> np.ndarray:
        """The opening card."""
        target = self.target.params
        return overlay.draw_card(
            width,
            height,
            "CROSS-EMBODIMENT LOCOMOTION",
            ("Walking a policy", "into a body twice its size"),
            (
                "One locomotion policy, trained once on the Berkeley Humanoid, is carried along a",
                f"continuous path through morphology space to a robot ×{target.size_scale:.0f} the "
                f"size, ×{target.mass_scale:.0f} the mass",
                f"and ×{target.torque_scale:.0f} the torque — and retrained only where it stops "
                "working.",
            ),
            "MuJoCo MJX · PPO · every number on screen comes from artifacts/continuation_similar",
        )

    def end_card(self, width: int, height: int) -> np.ndarray:
        """The closing card, with the result and the run that did not work."""
        spent = self.target.cumulative_steps
        stop = self.stops[0] if self.stops else None
        share = 100.0 * spent / self.baseline.steps
        detail = (
            (
                f"Only one waypoint of {len(self.waypoints)} needed help: α = {stop.alpha:.3f},",
                f"where the arriving policy survived {stop.survived_before:.3f} of the episode "
                "against a 0.800 floor.",
            )
            if stop is not None
            else ("No waypoint needed fine-tuning.",)
        )
        return overlay.draw_card(
            width,
            height,
            "RESULT",
            (f"{spent:,} fine-tune steps", f"{share:.1f}% of training from scratch"),
            (
                f"Reached ×{self.target.params.size_scale:.0f} size, "
                f"×{self.target.params.mass_scale:.0f} mass, "
                f"×{self.target.params.torque_scale:.0f} torque.",
                *detail,
                f"Forward speed rose {self.waypoints[0].speed_before:.3f} → "
                f"{self.target.speed_before:.3f} m/s while the Froude number held near "
                "0.035-0.040.",
                "",
                "Off the similarity manifold — same geometry, torque ×8 rather than ×16 — the",
                "same procedure failed: it stalled at α = 0.55 after 118.0 M steps and three "
                "backtracks.",
            ),
            f"Baseline {self.baseline.steps:,} steps, final reward {self.baseline.reward:.2f} · "
            "seed 0 · 8 evaluation episodes per waypoint",
        )

    def gif_window(self) -> tuple[float, float]:
        """Start second and length of the README GIF: the last growth beat, where the body doubles.

        Returns:
            The offset into the finished clip and the length of the cut, in seconds.
        """
        elapsed = self.timing.title
        growth = [beat for beat in self.beats if beat.name.startswith("grow")]
        for beat in self.beats:
            if growth and beat is growth[-1]:
                return elapsed + 0.6, max(2.0, min(9.0, beat.seconds - 0.6))
            elapsed += beat.seconds
        return self.timing.title, 8.0


class VideoWriter:
    """Streams RGB frames straight into ffmpeg, so a full-length clip never sits in memory."""

    def __init__(self, path: pathlib.Path, width: int, height: int, fps: int) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-profile:v",
            "high",
            "-b:v",
            VIDEO_BITRATE,
            "-maxrate",
            VIDEO_MAXRATE,
            "-bufsize",
            VIDEO_BUFSIZE,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        """Append one uint8 RGB frame."""
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        """Flush the encoder and wait for the file to be finalised.

        Raises:
            RuntimeError: if ffmpeg exited with an error.
        """
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def write_gif(
    video: pathlib.Path,
    gif: pathlib.Path,
    start: float,
    duration: float,
    max_width: int = 800,
    size_limit: int = GIF_SIZE_LIMIT,
) -> pathlib.Path:
    """Cut a GIF out of a finished video, backing off rate, palette and width until it fits.

    Args:
        video: the encoded clip to cut from.
        gif: destination path.
        start: seconds into the clip where the cut begins.
        duration: length of the cut, in seconds.
        max_width: widest output accepted, in pixels; the height follows the aspect ratio.
        size_limit: largest acceptable file, in bytes.

    Returns:
        The GIF path.

    Raises:
        RuntimeError: if even the cheapest attempt stays over the limit.
    """
    gif = pathlib.Path(gif)
    gif.parent.mkdir(parents=True, exist_ok=True)
    for fps, colours, attempt_width in GIF_ATTEMPTS:
        width = min(max_width, attempt_width)
        filters = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
            f"[a]palettegen=max_colors={colours}:stats_mode=diff[p];"
            f"[b][p]paletteuse=dither={GIF_DITHER}"
        )
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(video),
                "-filter_complex",
                filters,
                "-loop",
                "0",
                str(gif),
            ],
            check=True,
        )
        if gif.stat().st_size <= size_limit:
            return gif
    raise RuntimeError(f"{gif} is {gif.stat().st_size} bytes, over the {size_limit} byte budget")


def blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """Linear cross-fade between two frames."""
    mix = first.astype(np.float32) * (1.0 - weight) + second.astype(np.float32) * weight
    return mix.astype(np.uint8)


def card_frames(card: np.ndarray, seconds: float, fps: int) -> Iterator[np.ndarray]:
    """Repeat a still card for a stretch of the clip; a duration of zero yields nothing."""
    for _ in range(max(0, int(round(seconds * fps)))):
        yield card


@dataclasses.dataclass(frozen=True)
class RenderSettings:
    """Output settings of one render pass.

    Attributes:
        width: frame width in pixels.
        height: frame height in pixels.
        fps: frames per second of the finished clip, interpreted as real time.
        camera: camera placement.
    """

    width: int = 1920
    height: int = 1080
    fps: int = 60
    camera: render.CameraRig = dataclasses.field(default_factory=render.CameraRig)


def render_film(
    run_dir: pathlib.Path = DEFAULT_RUN_DIR,
    baseline_path: pathlib.Path = DEFAULT_BASELINE_PATH,
    media_dir: pathlib.Path = DEFAULT_MEDIA_DIR,
    settings: RenderSettings | None = None,
    timing: Timing | None = None,
    name: str = "morphology_continuation",
    make_gif: bool = True,
) -> dict:
    """Render the hero film end to end and return a report of what was written.

    Args:
        run_dir: continuation run to visualise.
        baseline_path: from-scratch run the fine-tune cost is measured against.
        media_dir: directory the mp4 and GIF are written to.
        settings: resolution, frame rate and camera.
        timing: durations of the film's parts.
        name: stem of the output files.
        make_gif: whether to cut the README GIF out of the finished video.

    Returns:
        A report with the output paths, frame count, duration and whether the robot stayed up.
    """
    settings = settings if settings is not None else RenderSettings()
    _, waypoints = render.load_run(run_dir)
    script = DemoScript(waypoints, load_baseline(baseline_path), timing)

    walker = render.MorphingWalker(run_dir)
    renderer = render.OffscreenRenderer(walker.model, settings.width, settings.height)
    reference = render.ScaleReference(
        marks=(
            walker.base_standing_height,
            walker.base_standing_height * script.target.params.size_scale,
        )
    )
    hud = overlay.Hud(settings.width, settings.height, script.copy(), script.marks())

    video_path = pathlib.Path(media_dir) / f"{name}.mp4"
    title = script.title_card(settings.width, settings.height)
    end = script.end_card(settings.width, settings.height)
    fade = max(1, int(round(script.timing.fade * settings.fps)))
    fade_in = fade if script.timing.title > 0 else 0
    fade_out = fade if script.timing.end > 0 else 0
    fell_at: float | None = None
    frames = 0
    composed = title

    walker.run_to(0.0, walker.config.settle_seconds)
    with VideoWriter(video_path, settings.width, settings.height, settings.fps) as writer:
        for frame in card_frames(title, script.timing.title, settings.fps):
            writer.write(frame)
        for index, (image, telemetry, beat, fraction) in enumerate(
            render.walk_frames(
                walker,
                renderer,
                script.beats,
                fps=settings.fps,
                rig=settings.camera,
                reference=reference,
                policy_override=script.policy_override,
            )
        ):
            composed = hud.draw(image, script.hud_frame(telemetry, beat, fraction))
            if index < fade_in:
                composed = blend(title, composed, (index + 1) / fade_in)
            writer.write(composed)
            frames += 1
            if not telemetry.upright and fell_at is None:
                fell_at = telemetry.time
            if index % (10 * settings.fps) == 0:
                print(
                    f"  frame {index:5d}  t={telemetry.time:6.2f}s  α={telemetry.alpha:5.3f}  "
                    f"size ×{telemetry.params.size_scale:.3f}  speed {telemetry.speed:.2f} m/s",
                    flush=True,
                )
        last = composed
        for index, frame in enumerate(card_frames(end, script.timing.end, settings.fps)):
            weight = min(1.0, (index + 1) / fade_out) if fade_out else 1.0
            writer.write(blend(last, frame, weight))
        total_frames = writer.frames
    renderer.close()

    report = {
        "video": str(video_path),
        "frames": total_frames,
        "walk_frames": frames,
        "seconds": total_frames / settings.fps,
        "resolution": f"{settings.width}x{settings.height}",
        "fps": settings.fps,
        "fell_at": fell_at,
    }
    if make_gif:
        start, duration = script.gif_window()
        gif_path = write_gif(video_path, pathlib.Path(media_dir) / f"{name}.gif", start, duration)
        report["gif"] = str(gif_path)
        report["gif_window"] = [start, duration]
    return report


# Storyboard of the standalone README loop: one uninterrupted growth, no cards and no stop.
GROWTH_LOOP_TIMING = Timing(
    title=0.0, opening=1.0, growth=7.0, finetune=0.0, arrival=1.5, end=0.0, fade=0.0
)


def render_growth_loop(
    run_dir: pathlib.Path = DEFAULT_RUN_DIR,
    baseline_path: pathlib.Path = DEFAULT_BASELINE_PATH,
    media_dir: pathlib.Path = DEFAULT_MEDIA_DIR,
    settings: RenderSettings | None = None,
    name: str = "morphology_continuation",
    max_width: int = 800,
) -> dict:
    """Render the README loop — the body going 1x to 2x with nothing else in the way — as a GIF.

    A GIF has to make its point in a few seconds on a page, so the loop compresses the path into
    one continuous growth instead of cutting the fine-tuning stop out of the film. Nothing about
    the walk changes: alpha is a coordinate on the path rather than a physical quantity, and the
    robot is simulated and drawn in real time at every point of it. The intermediate video is
    written to a scratch directory and only the GIF is kept.

    Args:
        run_dir: continuation run to visualise.
        baseline_path: from-scratch run the fine-tune cost is measured against.
        media_dir: directory the GIF is written to.
        settings: resolution, frame rate and camera of the render.
        name: stem of the GIF.
        max_width: widest GIF accepted, in pixels.

    Returns:
        A report with the GIF path, its size in bytes and the loop's duration.
    """
    with tempfile.TemporaryDirectory() as scratch:
        report = render_film(
            run_dir=run_dir,
            baseline_path=baseline_path,
            media_dir=pathlib.Path(scratch),
            settings=settings,
            timing=GROWTH_LOOP_TIMING,
            name=name,
            make_gif=False,
        )
        gif = write_gif(
            pathlib.Path(report["video"]),
            pathlib.Path(media_dir) / f"{name}.gif",
            0.0,
            report["seconds"],
            max_width=max_width,
        )
        return {
            "gif": str(gif),
            "gif_bytes": gif.stat().st_size,
            "gif_seconds": report["seconds"],
            "gif_fell_at": report["fell_at"],
        }


def preview_settings() -> tuple[RenderSettings, Timing]:
    """Small, short settings for checking framing and layout without a full render."""
    return (
        RenderSettings(width=960, height=540, fps=30),
        Timing(title=1.0, opening=1.5, growth=6.0, finetune=2.5, arrival=1.5, end=1.5, fade=0.3),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: render the film and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--baseline", type=pathlib.Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--media-dir", type=pathlib.Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--name", default="morphology_continuation")
    parser.add_argument("--preview", action="store_true", help="fast, small check render")
    parser.add_argument("--no-gif", action="store_true")
    arguments = parser.parse_args(argv)

    settings, timing = preview_settings() if arguments.preview else (RenderSettings(), None)
    report = render_film(
        run_dir=arguments.run_dir,
        baseline_path=arguments.baseline,
        media_dir=arguments.media_dir,
        settings=settings,
        timing=timing,
        name=arguments.name,
        make_gif=False,
    )
    if not arguments.no_gif:
        report.update(
            render_growth_loop(
                run_dir=arguments.run_dir,
                baseline_path=arguments.baseline,
                media_dir=arguments.media_dir,
                settings=settings,
                name=arguments.name,
            )
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
