import ffmpeg
from pathlib import Path
from typing import Tuple, Optional


class VideoConverter:
    """
    Converts uploaded videos to a normalized format for ML processing.
    
    Target specs:
    - Max resolution: 1920x1080 (1080p)
    - Max framerate: 30fps
    - Codec: H.264 for compatibility
    """
    
    MAX_WIDTH = 1920
    MAX_HEIGHT = 1080
    MAX_FPS = 30
    
    def __init__(self):
        pass
    
    def get_video_info(self, input_path: Path) -> dict:
        """Extract video metadata using ffprobe"""
        try:
            probe = ffmpeg.probe(str(input_path))
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Parse framerate (can be "30/1" or "29.97" format)
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den else 30
            else:
                fps = float(fps_str)
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': fps,
                'duration': float(probe['format'].get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown')
            }
        except Exception as e:
            raise ValueError(f"Failed to probe video: {str(e)}")
    
    def needs_conversion(self, input_path: Path) -> Tuple[bool, dict]:
        """
        Check if video needs conversion.
        
        Returns:
            Tuple of (needs_conversion, video_info)
        """
        info = self.get_video_info(input_path)
        
        needs_resize = info['width'] > self.MAX_WIDTH or info['height'] > self.MAX_HEIGHT
        needs_fps_reduction = info['fps'] > self.MAX_FPS
        
        return (needs_resize or needs_fps_reduction, info)
    
    def convert(
        self, 
        input_path: Path, 
        output_path: Optional[Path] = None,
        target_fps: Optional[float] = None
    ) -> Path:
        """
        Convert video to normalized format if needed.
        
        Args:
            input_path: Source video path
            output_path: Destination path (auto-generated if None)
            target_fps: Override target FPS (uses min of this and MAX_FPS)
            
        Returns:
            Path to converted video (may be same as input if no conversion needed)
        """
        needs_conv, info = self.needs_conversion(input_path)
        
        if not needs_conv:
            return input_path
        
        # Generate output path if not provided
        if output_path is None:
            suffix = input_path.suffix or '.mp4'
            output_path = input_path.with_stem(f"{input_path.stem}_converted").with_suffix('.mp4')
        
        # Calculate target dimensions maintaining aspect ratio
        scale_filter = None
        if info['width'] > self.MAX_WIDTH or info['height'] > self.MAX_HEIGHT:
            # Scale down while maintaining aspect ratio
            scale_filter = f"scale='min({self.MAX_WIDTH},iw)':min'({self.MAX_HEIGHT},ih)':force_original_aspect_ratio=decrease"
        
        # Determine target FPS
        fps = min(info['fps'], self.MAX_FPS)
        if target_fps:
            fps = min(fps, target_fps)
        
        # Build ffmpeg command
        stream = ffmpeg.input(str(input_path))
        
        # Apply filters
        if scale_filter:
            stream = stream.filter('scale', self.MAX_WIDTH, -2)  # -2 maintains aspect ratio with even number
        
        stream = stream.filter('fps', fps=fps)
        
        # Output with H.264 codec
        stream = ffmpeg.output(
            stream,
            str(output_path),
            vcodec='libx264',
            acodec='aac',
            preset='fast',
            crf=23  # Good quality/size balance
        )
        
        # Run conversion
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        return output_path
    
    async def convert_async(
        self, 
        input_path: Path, 
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Async wrapper for convert (runs in thread pool).
        Use this in async contexts.
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.convert,
            input_path,
            output_path
        )
