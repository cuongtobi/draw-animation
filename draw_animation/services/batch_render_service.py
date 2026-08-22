from __future__ import annotations

from pathlib import Path
from typing import Callable

from draw_animation.config import RenderConfig
from draw_animation.models import RenderJobResult
from draw_animation.services.image_discovery_service import ImageDiscoveryService
from draw_animation.services.video_render_service import RenderCancelled, VideoRenderService

ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]


class BatchRenderService:
    def __init__(
        self,
        discovery_service: ImageDiscoveryService | None = None,
        renderer: VideoRenderService | None = None,
    ) -> None:
        self._discovery = discovery_service or ImageDiscoveryService()
        self._renderer = renderer or VideoRenderService()

    def render_folder(
        self,
        input_folder: Path,
        output_folder: Path | None,
        config: RenderConfig,
        progress: ProgressCallback | None = None,
        log: LogCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> list[RenderJobResult]:
        config.validate()
        input_folder = Path(input_folder)
        output_folder = Path(output_folder) if output_folder else input_folder / "output"
        images = self._discovery.discover(input_folder)
        if not images:
            raise ValueError(f"No supported images found in: {input_folder}")
        output_folder.mkdir(parents=True, exist_ok=True)

        self._log(log, f"Found {len(images)} image(s)")
        self._log(log, f"Output folder: {output_folder}")
        results: list[RenderJobResult] = []
        total = len(images)

        for index, image_path in enumerate(images):
            if cancel_check is not None and cancel_check():
                self._log(log, "Batch cancelled before next image")
                break

            output_path = self._discovery.output_path(image_path, output_folder)
            self._log(log, f"[{index + 1}/{total}] Rendering {image_path.name} -> {output_path.name}")

            def local_progress(value: float, message: str) -> None:
                overall = (index + value) / total
                if progress is not None:
                    progress(overall, f"{image_path.name}: {message}")

            try:
                self._renderer.render(
                    image_path=image_path,
                    output_path=output_path,
                    config=config,
                    progress=local_progress,
                    cancel_check=cancel_check,
                )
            except RenderCancelled as exc:
                self._log(log, str(exc))
                break
            except Exception as exc:
                error = str(exc)
                results.append(RenderJobResult(image_path, output_path, False, error))
                self._log(log, f"ERROR {image_path.name}: {error}")
            else:
                results.append(RenderJobResult(image_path, output_path, True))
                self._log(log, f"DONE {output_path}")

        if progress is not None and results and all(result.success for result in results):
            progress(1.0, "Batch complete")
        return results

    @staticmethod
    def _log(callback: LogCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)
