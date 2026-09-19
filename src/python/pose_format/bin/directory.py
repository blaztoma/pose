import argparse
from pathlib import Path
from pose_format.bin.pose_estimation import pose_video, parse_additional_config
from pose_format.estimation.base import add_estimator_arguments, create_estimator
from typing import List
from tqdm import tqdm
from pose_format.bin.estimation_progress import in_worker, parallel_results, send_event, status
from pose_format.bin.filter_poses import filtered_pose_path, write_filtered_pose
import os
from functools import partial

# Note: untested other than .mp4. Support for .webm may have issues: https://github.com/sign-language-processing/pose/pull/126
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"]


def find_videos_with_missing_pose_files(
    directory: Path,
    video_suffixes: List[str] = None,
    recursive: bool = False,
    keep_video_suffixes: bool = False,
    output_directory: Path = None,
    filter_output: bool = False,
) -> List[Path]:
    """
    Finds videos with missing original or requested filtered .pose files.

    Parameters
    ----------
    directory: Path,
        Directory to search for videos in.
    video_suffixes:  List[str], optional
        Suffixes to look for, e.g. [".mp4", ".webm"]. If None, will use _SUPPORTED_VIDEO_FORMATS
    recursive: bool, optional
        Whether to look for video files recursively, or just the top-level. Defaults to false.
    keep_video_suffixes: bool, optional
        If true, when checking will append .pose suffix (e.g. foo.mp4->foo.mp4.pose, foo.webm->foo.webm.pose),
        If false, will replace it (foo.mp4 becomes foo.pose, and foo.webm ALSO becomes foo.pose).
        Default is false, which can cause name collisions.

    Returns
    -------
    List[Path]
        List of video paths without corresponding .pose files.
    """

    # Prevents the common gotcha with mutable default arg lists:
    # https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
    if video_suffixes is None:
        video_suffixes = SUPPORTED_VIDEO_FORMATS

    glob_method = getattr(directory, "rglob" if recursive else "glob")
    all_files = list(glob_method(f"*"))
    if isinstance(video_suffixes, str):
        video_suffixes = [video_suffixes]
    video_files = [path for path in all_files if path.is_file() and path.suffix.lower() in video_suffixes]

    videos_with_missing_pose_files = []

    for vid_path in video_files:
        corresponding_pose = get_corresponding_pose_path(vid_path, keep_video_suffixes, directory, output_directory)
        if (not corresponding_pose.is_file() or
                (filter_output and not filtered_pose_path(corresponding_pose).is_file())):
            videos_with_missing_pose_files.append(vid_path)

    return videos_with_missing_pose_files


def get_corresponding_pose_path(video_path: Path, keep_video_suffixes: bool = False,
                                directory: Path = None, output_directory: Path = None) -> Path:
    """
    Given a video path, and whether to keep the suffix, returns the expected corresponding path with .pose extension.

    Parameters
    ----------
    video_path : Path
        Path to a video file
    keep_video_suffixes : bool, optional
        Whether to keep suffix (e.g. foo.mp4 -> foo.mp4.pose)
        or replace (foo.mp4->foo.pose). Defaults to replace.

    Returns
    -------
    Path
        pathlib Path
    """
    if output_directory is not None:
        video_path = output_directory / video_path.relative_to(directory)
    if keep_video_suffixes:
        return video_path.with_name(f"{video_path.name}.pose")
    return video_path.with_suffix(".pose")


def process_video(keep_video_suffixes: bool, pose_format: str, additional_config: dict, vid_path: Path,
                  *, backend='mediapipe-legacy', device='cpu', model=None,
                  directory=None, output_directory=None, progress=True, filter_output=False) -> bool:
    label = str(vid_path.relative_to(directory)) if directory is not None else vid_path.name

    def report(completed, total):
        send_event('frame', label, completed, total)

    try:
        pose_path = get_corresponding_pose_path(vid_path, keep_video_suffixes, directory, output_directory)
        pose = None
        estimated = False
        if pose_path.is_file():
            status(f"Skipping {vid_path}, corresponding .pose file already created.")
        else:
            status(f'Estimating {vid_path} with {backend} on {device}')
            # pose_video function expects string, and passes it unchanged to cv2.VideoCapture(input_path)
            # if you give cv2.VideoCapture(input_path) a Path it crashes on older versions.
            # https://github.com/opencv/opencv/issues/15731
            pose = pose_video(str(vid_path.resolve()), str(pose_path.resolve()), pose_format, additional_config, progress=progress,
                       backend=backend, device=device, model=model, verbose=False,
                       progress_callback=report if in_worker() else None,
                       progress_position=1, progress_label=label)
            estimated = True
        if filter_output:
            # pose_video has saved the original; reuse its result without reading
            # the potentially large file again. Refresh stale filtered output if
            # its original had to be regenerated.
            count = write_filtered_pose(pose_path, pose, overwrite=estimated)
            if count is not None:
                status(f'Filtered {pose_path} -> {filtered_pose_path(pose_path)} ({count} landmarks masked)')
        return True
            
    except (ValueError, RuntimeError, ImportError, OSError) as e:
        status(f"Error on {vid_path}: {type(e).__name__}: {e}")
    finally:
        if in_worker():
            send_event('finished', label)
    return False
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--format",
        choices=["mediapipe"],
        default="mediapipe",
        type=str,
        help="type of pose estimation to use",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="Directory to search for videos in",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Whether to search for videos recursively",
    )
    parser.add_argument(
        "--keep-video-suffixes",
        action="store_true",
        help="Whether to drop the video extension (output for foo.mp4 becomes foo.pose, and foo.webm ALSO becomes foo.pose) or append to it (foo.mp4 becomes foo.mp4.pose, foo.webm output is foo.webm.pose). If there are multiple videos with the same basename but different extensions, this will create a .pose file for each. Otherwise only the first video will be posed.",
    )
    parser.add_argument(
        "--video-suffixes",
        type=str,
        choices=SUPPORTED_VIDEO_FORMATS,
        default=SUPPORTED_VIDEO_FORMATS,
        help="Video extensions to search for. Defaults to searching for all supported.",
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=1, 
        help="Number of multiprocessing workers.", 
        required=False
    )
    parser.add_argument(
        "--additional-config",
        type=str,
        help="additional configuration for the pose estimator",
    )
    add_estimator_arguments(parser)
    parser.add_argument('--output-directory', type=Path,
                        help='Separate output root, preserving input subdirectories (existing outputs are skipped)')
    parser.add_argument('--no-progress', action='store_true', help='Disable video and frame progress bars')
    parser.add_argument('--filter', dest='filter_output', action='store_true',
                        help='Also save *_filtered.pose with unreliable leg points (hips preserved); '
                             'reuse existing originals without estimating again')
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f'Input directory does not exist: {args.directory}')
    if args.num_workers < 1:
        parser.error('--num-workers must be positive')
    additional_config = parse_additional_config(args.additional_config)
    videos_with_missing_pose_files = find_videos_with_missing_pose_files(
        args.directory,
        video_suffixes=args.video_suffixes,
        recursive=args.recursive,
        keep_video_suffixes=args.keep_video_suffixes,
        output_directory=args.output_directory,
        filter_output=args.filter_output,
    )

    # Filtering existing poses needs neither an estimator nor a model file.
    if any(not get_corresponding_pose_path(video, args.keep_video_suffixes,
                                          args.directory, args.output_directory).is_file()
           for video in videos_with_missing_pose_files):
        try:
            create_estimator(args.backend, args.device, args.model, additional_config)
        except (ValueError, OSError) as error:
            parser.error(str(error))

    print(f"Found {len(videos_with_missing_pose_files)} videos missing requested pose files.")

    pose_files_that_will_be_created = {get_corresponding_pose_path(vid_path, args.keep_video_suffixes) for vid_path in videos_with_missing_pose_files}

    if len(pose_files_that_will_be_created) < len(videos_with_missing_pose_files):
        continue_input = input(
            f"With current naming strategy (without --keep-video-suffixes), name collisions will result in only {len(pose_files_that_will_be_created)} .pose files being created. Continue? [y/n]"
        )
        if continue_input.lower() != "y":
            print(f"Exiting. To keep video suffixes and avoid collisions, use --keep-video-suffixes")
            exit()

    pose_with_no_errors_count = 0

    if args.num_workers == 1:
        print('Process sequentially ...')
    else:
        available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        print(f'Multiprocessing with {args.num_workers} workers on {available_cpus} available CPUs ...')

    func = partial(process_video, args.keep_video_suffixes, args.format, additional_config,
                   backend=args.backend, device=args.device, model=args.model,
                   directory=args.directory, output_directory=args.output_directory,
                   progress=not args.no_progress, filter_output=args.filter_output)
    results = (map(func, videos_with_missing_pose_files)
               if args.num_workers == 1 else
               parallel_results(func, videos_with_missing_pose_files, args.num_workers))
    for success in tqdm(results, total=len(videos_with_missing_pose_files), desc='Videos',
                        unit='video', position=0, dynamic_ncols=True, disable=args.no_progress):
        if success:
            pose_with_no_errors_count += 1

    print(f"Successfully created pose files for {pose_with_no_errors_count}/{len(videos_with_missing_pose_files)} video files")
    if pose_with_no_errors_count != len(videos_with_missing_pose_files):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
