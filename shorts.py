import requests
import re
import json
import sys
import concurrent.futures
import subprocess
import time
import os
import math
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any

# mulai
def extract_youtube_id(url):
    regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_shorts(video_id):
    url = f"https://secondary.api.2short.ai/shorts?youtubeVideoId={video_id}&language=auto"
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://app.2short.ai/',
        'Origin': 'https://app.2short.ai',
        'Connection': 'keep-alive'
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, verify=True)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def normalize_timings_based_on_first_transcript(transcript_array, emojis_data):
    if not transcript_array:
        return transcript_array, emojis_data
    try:
        first_start_time = None
        for item in transcript_array:
            if 'start' in item:
                try:
                    start_time = float(item['start'])
                    if first_start_time is None or start_time < first_start_time:
                        first_start_time = start_time
                except:
                    continue
        if first_start_time is None:
            return transcript_array, emojis_data
        normalized_transcript = []
        for item in transcript_array:
            normalized_item = item.copy()
            if 'start' in normalized_item:
                try:
                    original_start = float(normalized_item['start'])
                    normalized_item['start'] = f"{original_start - first_start_time:.1f}"
                except:
                    pass
            normalized_transcript.append(normalized_item)
        normalized_emojis = {}
        if emojis_data:
            for word, emoji_info in emojis_data.items():
                if 'start' in emoji_info and emoji_info['start']:
                    try:
                        original_emoji_start = float(emoji_info['start'])
                        normalized_start = original_emoji_start - first_start_time
                        if normalized_start >= 0:
                            normalized_emojis[word] = {
                                'emoji': emoji_info.get('emoji', ''),
                                'start': f"{normalized_start:.1f}"
                            }
                    except:
                        normalized_emojis[word] = emoji_info.copy()
                else:
                    normalized_emojis[word] = emoji_info.copy()
        return normalized_transcript, normalized_emojis
    except Exception as e:
        print(f"Error normalizing timings: {e}")
        return transcript_array, emojis_data

def retry_get_transcript_data(short_id, max_retries=3):
    transcript_url = f"https://secondary.api.2short.ai/shorts/{short_id}/transcript"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        "Accept": "application/json, text/plain, */*",
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(transcript_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            else:
                return None
    return None

def download_short_video(short_id, download_dir, max_retries=2):
    headers = {
        "Host": "secondary.api.2short.ai",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "If-None-Match": 'W/"5e8-xl5pJN1VBgD7HreiWl9UyP2lAM8"',
        "Te": "trailers",
        "Connection": "keep-alive"
    }
    for attempt in range(max_retries):
        try:
            url1 = f"https://secondary.api.2short.ai/shorts/{short_id}"
            r1 = requests.get(url1, headers=headers)
            r1.raise_for_status()
            token_data = r1.json()
            token = token_data.get('clipRequestToken')
            if not token:
                print(f"Token tidak ditemukan untuk {short_id}, retry {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            url2 = f"https://secondary.clips.api.2short.ai/?token={token}"
            r2 = requests.get(url2, timeout=60)
            r2.raise_for_status()
            clip_url = None
            for line in r2.text.split('\n'):
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'clipUrl' in data:
                            clip_url = data['clipUrl']
                            break
                    except:
                        continue
            if not clip_url:
                print(f"clipUrl tidak ditemukan untuk {short_id}, retry {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            filename = os.path.join(download_dir, f"{short_id}.mp4")
            result = subprocess.run(
                ["curl", "-s", "-f", "-o", filename, clip_url],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                return True
            else:
                print(f"Download gagal untuk {short_id}: {result.stderr[:100]}, retry {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return False
        except Exception as e:
            print(f"Error download {short_id}: {str(e)[:100]}, retry {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return False
    return False

def get_emojis_with_retry(emojis_url, short_id, max_retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        "Accept": "application/json, text/plain, */*",
    }
    for attempt in range(max_retries):
        try:
            emoji_response = requests.get(emojis_url, headers=headers, timeout=30)
            if emoji_response.status_code != 200:
                print(f"EmojisURL invalid (HTTP {emoji_response.status_code}) untuk {short_id}, coba URL baru...")
                transcript_data = retry_get_transcript_data(short_id)
                if transcript_data and 'short' in transcript_data:
                    new_emojis_url = transcript_data['short'].get('emojisUrl')
                    if new_emojis_url and new_emojis_url != emojis_url:
                        print(f"Mendapatkan emojisUrl baru untuk {short_id}")
                        emojis_url = new_emojis_url
                        continue
            emoji_response.raise_for_status()
            emojis_raw = emoji_response.json()
            if not isinstance(emojis_raw, list):
                print(f"Format emoji tidak valid untuk {short_id}, coba URL baru...")
                transcript_data = retry_get_transcript_data(short_id)
                if transcript_data and 'short' in transcript_data:
                    new_emojis_url = transcript_data['short'].get('emojisUrl')
                    if new_emojis_url and new_emojis_url != emojis_url:
                        emojis_url = new_emojis_url
                        continue
            emojis_data = {}
            for item in emojis_raw:
                if isinstance(item, dict) and 'word' in item:
                    word = item['word']
                    emojis_data[word] = {
                        'emoji': item.get('emoji', ''),
                        'start': item.get('start', '')
                    }
            return emojis_data, emojis_url
        except Exception as e:
            print(f"Error ambil emoji {short_id}: {str(e)[:100]}, attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                if attempt == 0:
                    transcript_data = retry_get_transcript_data(short_id)
                    if transcript_data and 'short' in transcript_data:
                        new_emojis_url = transcript_data['short'].get('emojisUrl')
                        if new_emojis_url and new_emojis_url != emojis_url:
                            emojis_url = new_emojis_url
                time.sleep(1 * (attempt + 1))
                continue
            else:
                return None, emojis_url
    return None, emojis_url

def process_single_short(short):
    short_id = short.get('_id')
    if not short_id:
        return None
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        "Accept": "application/json, text/plain, */*",
    }
    transcript_data = None
    for attempt in range(3):
        try:
            transcript_url = f"https://secondary.api.2short.ai/shorts/{short_id}/transcript"
            response = requests.get(transcript_url, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"Transcript URL invalid (HTTP {response.status_code}) untuk {short_id}, retry...")
                time.sleep(2)
                continue
            response.raise_for_status()
            transcript_data = response.json()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            else:
                print(f"Gagal ambil transkrip untuk {short_id} setelah 3 percobaan")
                return None
    if not transcript_data:
        return None
    transcript_array = None
    emojis_url = None
    if 'short' in transcript_data:
        short_data = transcript_data['short'].copy()
        if 'transcript' in short_data:
            transcript_array = short_data['transcript']
        if 'emojisUrl' in short_data:
            emojis_url = short_data['emojisUrl']
    emojis_data = None
    if emojis_url:
        emojis_data, final_emojis_url = get_emojis_with_retry(emojis_url, short_id)
        if final_emojis_url != emojis_url and emojis_data:
            print(f"Berhasil update emojisUrl untuk {short_id}")
    normalized_transcript, normalized_emojis = normalize_timings_based_on_first_transcript(
        transcript_array, emojis_data
    )
    return {
        'id': short_id,
        'title': short.get('title', ''),
        'viralityScore': short.get('viralityScore', 0),
        'transcript': normalized_transcript,
        'emojis': normalized_emojis
    }

class FacePlusPlusDetector:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = "https://api-us.faceplusplus.com/facepp/v3/detect"

    def detect_faces(self, image_path):
        try:
            files = {'image_file': open(image_path, 'rb')}
            data = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'return_landmark': 0,
            }
            response = requests.post(self.api_url, files=files, data=data)
            files['image_file'].close()
            result = response.json()
            if 'faces' in result:
                return self._format_results(result, image_path)
            else:
                print(f"    Tidak ada wajah yang terdeteksi")
                return None
        except Exception as e:
            print(f"    Error deteksi wajah: {str(e)}")
            return None

    def _format_results(self, api_result, image_path):
        formatted_result = {"faces": []}
        for face in api_result.get('faces', []):
            face_rectangle = face.get('face_rectangle', {})
            face_data = {
                "x": face_rectangle.get('left', 0),
                "y": face_rectangle.get('top', 0),
                "width": face_rectangle.get('width', 0),
                "height": face_rectangle.get('height', 0)
            }
            formatted_result["faces"].append(face_data)
        return formatted_result

class VideoFaceAnalyzer:
    def __init__(self, api_key, api_secret, temp_dir):
        self.face_detector = FacePlusPlusDetector(api_key, api_secret)
        self.temp_dir = temp_dir

    def run_ffmpeg_silent(self, cmd):
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    def get_video_duration(self, input_video):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_video], capture_output=True, text=True)
        return float(result.stdout.strip()) * 1000

    def detect_scene_changes(self, input_video, threshold=0.4):
        cmd = ["ffmpeg", "-i", input_video, "-filter_complex", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        timestamps_ms = [0.0]
        for line in process.stderr:
            if "pts_time:" in line:
                match = re.search(r"pts_time:([0-9.]+)", line)
                if match:
                    t_seconds = float(match.group(1))
                    t_ms = t_seconds * 1000
                    timestamps_ms.append(t_ms)
        process.wait()
        return sorted(set(timestamps_ms))

    def capture_scene_screenshots(self, input_video, scenes):
        screenshots_dir = os.path.join(self.temp_dir, "screenshots")
        Path(screenshots_dir).mkdir(exist_ok=True)
        for scene in scenes:
            middle_point_seconds = (scene["start"] + scene["end"]) / 2
            output_filename = f"{scene['scenes']}.jpg"
            output_path = os.path.join(screenshots_dir, output_filename)
            cmd = ["ffmpeg", "-ss", str(middle_point_seconds), "-i", input_video, "-vframes", "1", "-q:v", "2", "-vf", "scale='min(1920,iw)':-2", "-y", output_path]
            print(f"  Mengambil screenshot scene {scene['scenes']}...")
            result = self.run_ffmpeg_silent(cmd)
            if result.returncode == 0 and os.path.exists(output_path):
                scene["screenshot_path"] = output_path
                print(f"    Disimpan: {output_filename}")
            else:
                print(f"    Gagal mengambil screenshot")
                scene["screenshot_path"] = None
        return scenes

    def analyze_faces_in_scenes(self, scenes):
        print("\n[*] Menganalisis wajah dalam setiap scene...")
        for i, scene in enumerate(scenes):
            screenshot_path = scene.get("screenshot_path")
            if not screenshot_path or not os.path.exists(screenshot_path):
                print(f"\nScene {scene['scenes']}: Tidak ada screenshot, dilewati")
                scene["face"] = []
                continue
            print(f"\nScene {scene['scenes']}: Analisis wajah...")
            face_results = self.face_detector.detect_faces(screenshot_path)
            if face_results:
                scene["face"] = face_results["faces"]
                num_faces = len(face_results["faces"])
                print(f"    Ditemukan {num_faces} wajah")
                for j, face in enumerate(face_results["faces"]):
                    print(f"      Wajah {j+1}: posisi ({face['x']}, {face['y']}), ukuran {face['width']}x{face['height']}")
            else:
                scene["face"] = []
                print(f"    Tidak ada wajah terdeteksi")
            if i < len(scenes) - 1:
                time.sleep(0.5)
        return scenes

    def ms_to_seconds(self, ms_value):
        return round(ms_value / 1000.0, 6)

    def analyze_video(self, input_video, scene_threshold=0.4):
        print(f"[*] Menganalisis video: {input_video}")
        print("[*] Mendapatkan durasi video...")
        duration_ms = self.get_video_duration(input_video)
        print("[*] Mendeteksi perubahan scene...")
        timestamps_ms = self.detect_scene_changes(input_video, scene_threshold)
        timestamps_ms.append(duration_ms)
        timestamps_seconds = [self.ms_to_seconds(ts) for ts in timestamps_ms]
        scenes = []
        for i in range(len(timestamps_seconds) - 1):
            scene = {
                "scenes": i + 1,
                "start": timestamps_seconds[i],
                "end": timestamps_seconds[i + 1],
            }
            scenes.append(scene)
        print(f"[*] Terdeteksi {len(scenes)} scene")
        print("[*] Mengambil screenshot setiap scene...")
        scenes = self.capture_scene_screenshots(input_video, scenes)
        scenes = self.analyze_faces_in_scenes(scenes)
        for scene in scenes:
            if "screenshot_path" in scene:
                del scene["screenshot_path"]
        screenshots_dir = os.path.join(self.temp_dir, "screenshots")
        if os.path.exists(screenshots_dir):
            shutil.rmtree(screenshots_dir)
        result_data = {"scenes": scenes}
        scenes_with_faces = sum(1 for scene in scenes if len(scene.get("face", [])) > 0)
        total_faces = sum(len(scene.get("face", [])) for scene in scenes)
        print(f"\n[✓] Analisis selesai!")
        print(f"\nStatistik (hanya untuk info):")
        print(f"  • Total scene: {len(scenes)}")
        print(f"  • Scene dengan wajah: {scenes_with_faces}")
        print(f"  • Total wajah terdeteksi: {total_faces}")
        return result_data

def run_command(cmd: List[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def make_even(value: int) -> int:
    if value % 2 == 1:
        value -= 1
    return value

def clamp(value: int, min_val: int, max_val: int) -> int:
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

def get_video_info(input_file: str) -> Tuple[int, int, int]:
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,bit_rate',
        '-of', 'csv=p=0',
        input_file
    ]
    info = run_command(cmd)
    parts = info.split(',')
    width = int(parts[0])
    height = int(parts[1])
    bitrate = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    return width, height, bitrate

def calculate_target_bitrate(original_bitrate: int, original_pixels: int, crop_pixels: int, 
                           min_bitrate: int = 3000000, max_bitrate: int = 15000000) -> int:
    if original_bitrate == 0:
        if original_pixels > 2000000:
            original_bitrate = 12000000
        elif original_pixels > 1000000:
            original_bitrate = 8000000
        else:
            original_bitrate = 4000000
    pixel_ratio = crop_pixels / original_pixels
    target_bitrate = int(original_bitrate * pixel_ratio * 1.2)
    if target_bitrate < min_bitrate:
        target_bitrate = min_bitrate
    elif target_bitrate > max_bitrate:
        target_bitrate = max_bitrate
    return target_bitrate

def build_filter_for_scene(scene_idx: int, scene_data: Dict, width: int, height: int,
                          target_width: int, target_height: int) -> str:
    start = scene_data['start']
    end = scene_data['end']
    faces = scene_data.get('face', [])
    face_count = len(faces)
    scene_filter = f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS"
    if face_count == 0:
        print(f"  Layout: 0 wajah -> crop tengah")
        crop_x = (width - target_width) // 2
        crop_y = 0
        crop_x = clamp(crop_x, 0, width - target_width)
        crop_x = make_even(crop_x)
        scene_filter += f",crop=w={target_width}:h={target_height}:x={crop_x}:y={crop_y}"
        scene_filter += f",pad=w={target_width}:h={target_height}:x=0:y=0:color=black[seg{scene_idx}]"
    elif face_count == 1:
        print(f"  Layout: 1 wajah (center)")
        face = faces[0]
        f_x = face['x']
        f_y = face['y']
        f_w = face['width']
        f_h = face['height']
        f_center_x = f_x + f_w // 2
        f_center_y = f_y + f_h // 2
        crop_x = f_center_x - target_width // 2
        crop_y = f_center_y - target_height // 2
        crop_x = clamp(crop_x, 0, width - target_width)
        crop_y = clamp(crop_y, 0, height - target_height)
        crop_x = make_even(crop_x)
        crop_y = make_even(crop_y)
        scene_filter += f",crop=w={target_width}:h={target_height}:x={crop_x}:y={crop_y}"
        scene_filter += f",pad=w={target_width}:h={target_height}:x=0:y=0:color=black[seg{scene_idx}]"
    elif face_count == 2:
        print(f"  Layout: 2 wajah (vertical stack)")
        face1 = faces[0]
        face2 = faces[1]
        f1_x = face1['x']; f1_y = face1['y']; f1_w = face1['width']; f1_h = face1['height']
        f2_x = face2['x']; f2_y = face2['y']; f2_w = face2['width']; f2_h = face2['height']
        f1_center_x = f1_x + f1_w // 2
        f1_center_y = f1_y + f1_h // 2
        f2_center_x = f2_x + f2_w // 2
        f2_center_y = f2_y + f2_h // 2
        if f1_center_y < f2_center_y:
            top_x, top_y = f1_center_x, f1_center_y
            bottom_x, bottom_y = f2_center_x, f2_center_y
        else:
            top_x, top_y = f2_center_x, f2_center_y
            bottom_x, bottom_y = f1_center_x, f1_center_y
        part_height = target_height // 2
        part_height = make_even(part_height)
        top_crop_x = top_x - target_width // 2
        top_crop_y = top_y - part_height // 2
        top_crop_x = clamp(top_crop_x, 0, width - target_width)
        top_crop_y = clamp(top_crop_y, 0, height - part_height)
        top_crop_x = make_even(top_crop_x)
        top_crop_y = make_even(top_crop_y)
        bottom_crop_x = bottom_x - target_width // 2
        bottom_crop_y = bottom_y - part_height // 2
        bottom_crop_x = clamp(bottom_crop_x, 0, width - target_width)
        bottom_crop_y = clamp(bottom_crop_y, 0, height - part_height)
        bottom_crop_x = make_even(bottom_crop_x)
        bottom_crop_y = make_even(bottom_crop_y)
        scene_filter += f",split=2[scene{scene_idx}_a][scene{scene_idx}_b];"
        scene_filter += f"[scene{scene_idx}_a]crop=w={target_width}:h={part_height}:x={top_crop_x}:y={top_crop_y}"
        scene_filter += f",pad=w={target_width}:h={part_height}:x=0:y=0:color=black"
        scene_filter += f",scale={target_width}:{part_height}[top{scene_idx}];"
        scene_filter += f"[scene{scene_idx}_b]crop=w={target_width}:h={part_height}:x={bottom_crop_x}:y={bottom_crop_y}"
        scene_filter += f",pad=w={target_width}:h={part_height}:x=0:y=0:color=black"
        scene_filter += f",scale={target_width}:{part_height}[bottom{scene_idx}];"
        scene_filter += f"[top{scene_idx}][bottom{scene_idx}]vstack"
        scene_filter += f",pad=w={target_width}:h={target_height}:x=0:y=0:color=black[seg{scene_idx}]"
    elif face_count == 3:
        print(f"  Layout: 3 wajah (2 atas, 1 bawah) - EQUAL HEIGHT PER FACE")
        face_height = target_height // 2
        top_width = target_width // 2
        face_height = make_even(face_height)
        top_width = make_even(top_width)
        print(f"    Each face height: {face_height} px")
        print(f"    Top row height: {face_height} px (2 faces)")
        print(f"    Bottom row height: {face_height} px (1 face)")
        face_centers = []
        for face in faces:
            f_x = face['x']; f_y = face['y']; f_w = face['width']; f_h = face['height']
            center_x = f_x + f_w // 2
            center_y = f_y + f_h // 2
            face_centers.append((center_x, center_y))
        indices = list(range(3))
        for i in range(2):
            for j in range(2 - i):
                idx1 = indices[j]
                idx2 = indices[j + 1]
                if face_centers[idx1][1] > face_centers[idx2][1]:
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]
        top1_idx, top2_idx, bottom_idx = indices
        top1_x, top1_y = face_centers[top1_idx]
        top2_x, top2_y = face_centers[top2_idx]
        bottom_x, bottom_y = face_centers[bottom_idx]
        top1_crop_x = top1_x - top_width // 2
        top1_crop_y = top1_y - face_height // 2
        top1_crop_x = clamp(top1_crop_x, 0, width - top_width)
        top1_crop_y = clamp(top1_crop_y, 0, height - face_height)
        top1_crop_x = make_even(top1_crop_x)
        top1_crop_y = make_even(top1_crop_y)
        top2_crop_x = top2_x - top_width // 2
        top2_crop_y = top2_y - face_height // 2
        top2_crop_x = clamp(top2_crop_x, 0, width - top_width)
        top2_crop_y = clamp(top2_crop_y, 0, height - face_height)
        top2_crop_x = make_even(top2_crop_x)
        top2_crop_y = make_even(top2_crop_y)
        bottom_crop_x = bottom_x - target_width // 2
        bottom_crop_y = bottom_y - face_height // 2
        bottom_crop_x = clamp(bottom_crop_x, 0, width - target_width)
        bottom_crop_y = clamp(bottom_crop_y, 0, height - face_height)
        bottom_crop_x = make_even(bottom_crop_x)
        bottom_crop_y = make_even(bottom_crop_y)
        scene_filter += f",split=3[scene{scene_idx}_a][scene{scene_idx}_b][scene{scene_idx}_c];"
        scene_filter += f"[scene{scene_idx}_a]crop=w={top_width}:h={face_height}:x={top1_crop_x}:y={top1_crop_y}"
        scene_filter += f",pad=w={top_width}:h={face_height}:x=0:y=0:color=black"
        scene_filter += f",scale={top_width}:{face_height}[top_left{scene_idx}];"
        scene_filter += f"[scene{scene_idx}_b]crop=w={top_width}:h={face_height}:x={top2_crop_x}:y={top2_crop_y}"
        scene_filter += f",pad=w={top_width}:h={face_height}:x=0:y=0:color=black"
        scene_filter += f",scale={top_width}:{face_height}[top_right{scene_idx}];"
        scene_filter += f"[scene{scene_idx}_c]crop=w={target_width}:h={face_height}:x={bottom_crop_x}:y={bottom_crop_y}"
        scene_filter += f",pad=w={target_width}:h={face_height}:x=0:y=0:color=black"
        scene_filter += f",scale={target_width}:{face_height}[bottom{scene_idx}];"
        scene_filter += f"[top_left{scene_idx}][top_right{scene_idx}]hstack"
        scene_filter += f",pad=w={target_width}:h={face_height}:x=0:y=0:color=black[top_combined{scene_idx}];"
        scene_filter += f"[top_combined{scene_idx}][bottom{scene_idx}]vstack"
        scene_filter += f",pad=w={target_width}:h={target_height}:x=0:y=0:color=black[seg{scene_idx}]"
    elif face_count == 4:
        print(f"  Layout: 4 wajah (grid 2x2)")
        face_width = target_width // 2
        face_height = target_height // 2
        face_width = make_even(face_width)
        face_height = make_even(face_height)
        face_centers = []
        for face in faces:
            f_x = face['x']; f_y = face['y']; f_w = face['width']; f_h = face['height']
            center_x = f_x + f_w // 2
            center_y = f_y + f_h // 2
            face_centers.append((center_x, center_y))
        tl_crop_x = face_centers[0][0] - face_width // 2
        tl_crop_y = face_centers[0][1] - face_height // 2
        tl_crop_x = clamp(tl_crop_x, 0, width - face_width)
        tl_crop_y = clamp(tl_crop_y, 0, height - face_height)
        tl_crop_x = make_even(tl_crop_x)
        tl_crop_y = make_even(tl_crop_y)
        tr_crop_x = face_centers[1][0] - face_width // 2
        tr_crop_y = face_centers[1][1] - face_height // 2
        tr_crop_x = clamp(tr_crop_x, 0, width - face_width)
        tr_crop_y = clamp(tr_crop_y, 0, height - face_height)
        tr_crop_x = make_even(tr_crop_x)
        tr_crop_y = make_even(tr_crop_y)
        bl_crop_x = face_centers[2][0] - face_width // 2
        bl_crop_y = face_centers[2][1] - face_height // 2
        bl_crop_x = clamp(bl_crop_x, 0, width - face_width)
        bl_crop_y = clamp(bl_crop_y, 0, height - face_height)
        bl_crop_x = make_even(bl_crop_x)
        bl_crop_y = make_even(bl_crop_y)
        br_crop_x = face_centers[3][0] - face_width // 2
        br_crop_y = face_centers[3][1] - face_height // 2
        br_crop_x = clamp(br_crop_x, 0, width - face_width)
        br_crop_y = clamp(br_crop_y, 0, height - face_height)
        br_crop_x = make_even(br_crop_x)
        br_crop_y = make_even(br_crop_y)
        scene_filter += f",split=4[scene{scene_idx}_a][scene{scene_idx}_b][scene{scene_idx}_c][scene{scene_idx}_d];"
        scene_filter += f"[scene{scene_idx}_a]crop=w={face_width}:h={face_height}:x={tl_crop_x}:y={tl_crop_y}"
        scene_filter += f",pad=w={face_width}:h={face_height}:x=0:y=0:color=black"
        scene_filter += f",scale={face_width}:{face_height}[tl{scene_idx}];"
        scene_filter += f"[scene{scene_idx}_b]crop=w={face_width}:h={face_height}:x={tr_crop_x}:y={tr_crop_y}"
        scene_filter += f",pad=w={face_width}:h={face_height}:x=0:y=0:color=black"
        scene_filter += f",scale={face_width}:{face_height}[tr{scene_idx}];"
        scene_filter += f"[scene{scene_idx}_c]crop=w={face_width}:h={face_height}:x={bl_crop_x}:y={bl_crop_y}"
        scene_filter += f",pad=w={face_width}:h={face_height}:x=0:y=0:color=black"
        scene_filter += f",scale={face_width}:{face_height}[bl{scene_idx}];"
        scene_filter += f"[scene{scene_idx}_d]crop=w={face_width}:h={face_height}:x={br_crop_x}:y={br_crop_y}"
        scene_filter += f",pad=w={face_width}:h={face_height}:x=0:y=0:color=black"
        scene_filter += f",scale={face_width}:{face_height}[br{scene_idx}];"
        scene_filter += f"[tl{scene_idx}][tr{scene_idx}]hstack"
        scene_filter += f",pad=w={target_width}:h={face_height}:x=0:y=0:color=black[top_row{scene_idx}];"
        scene_filter += f"[bl{scene_idx}][br{scene_idx}]hstack"
        scene_filter += f",pad=w={target_width}:h={face_height}:x=0:y=0:color=black[bottom_row{scene_idx}];"
        scene_filter += f"[top_row{scene_idx}][bottom_row{scene_idx}]vstack"
        scene_filter += f",pad=w={target_width}:h={target_height}:x=0:y=0:color=black[seg{scene_idx}]"
    else:
        print(f"  Layout: 0 wajah -> crop tengah (fallback)")
        crop_x = (width - target_width) // 2
        crop_y = 0
        crop_x = clamp(crop_x, 0, width - target_width)
        crop_x = make_even(crop_x)
        scene_filter += f",crop=w={target_width}:h={target_height}:x={crop_x}:y={crop_y}"
        scene_filter += f",pad=w={target_width}:h={target_height}:x=0:y=0:color=black[seg{scene_idx}]"
    return scene_filter

def convert_to_portrait(input_video, json_data, output_video, temp_dir):
    print("=== Portrait Converter: Maximum Quality ===")
    if not os.path.exists(input_video):
        print(f"Error: File '{input_video}' tidak ditemukan")
        return False
    width, height, original_bitrate = get_video_info(input_video)
    print(f"Original video: {width}x{height}")
    print(f"Original bitrate: {original_bitrate // 1000} kbps")
    crop_w = height * 9 // 16
    if crop_w % 2 == 1:
        crop_w -= 1
    target_width = crop_w
    target_height = height
    if target_width % 2 == 1:
        target_width += 1
    if target_height % 2 == 1:
        target_height += 1
    print(f"Target output size: {target_width}x{target_height} (9:16 portrait)")
    print("All scenes will be forced to this exact size")
    original_pixels = width * height
    crop_pixels = target_width * target_height
    target_bitrate = calculate_target_bitrate(
        original_bitrate, original_pixels, crop_pixels
    )
    print(f"Target bitrate: {target_bitrate // 1000} kbps")
    filter_complex = ""
    scenes = json_data['scenes']
    scene_count = len(scenes)
    for i, scene in enumerate(scenes):
        print(f"Building filter for scene {i+1}...")
        scene_filter = build_filter_for_scene(i, scene, width, height, target_width, target_height)
        filter_complex += f"{scene_filter};"
    filter_complex += f"[seg0]"
    for i in range(1, scene_count):
        filter_complex += f"[seg{i}]"
    filter_complex += f"concat=n={scene_count}:v=1:a=0[outv]"
    print("Filter complex built successfully")
    print(f"Total scenes: {scene_count}")
    print(f"All outputs forced to: {target_width}x{target_height}")
    print("\n=== Encoding dengan kualitas maksimal ===")
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '0:a',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '17',
        '-profile:v', 'high',
        '-level', '4.2',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-c:a', 'copy',
        output_video
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Gagal membuat output video")
        print(f"Error: {e}")
        return False
    if os.path.exists(output_video):
        print("Konversi berhasil!")
        print(f"Output: {output_video}")
        return True
    else:
        print("Gagal membuat output video")
        return False

def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    filename = ''.join(char for char in filename if ord(char) >= 32)
    if len(filename) > 200:
        filename = filename[:200]
    return filename

def read_youtube_urls_from_file():
    url_file = "url.txt"
    if not os.path.exists(url_file):
        print(f"File {url_file} tidak ditemukan")
        sys.exit(1)
    
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print(f"File {url_file} kosong atau tidak berisi URL yang valid")
        sys.exit(1)
    
    valid_urls = []
    for url in urls:
        video_id = extract_youtube_id(url)
        if video_id:
            valid_urls.append((url, video_id))
        else:
            print(f"URL tidak valid, dilewati: {url}")
    
    if not valid_urls:
        print("Tidak ada URL YouTube yang valid dalam file url.txt")
        sys.exit(1)
    
    return valid_urls

def main_pipeline():
    print("=" * 60)
    print("YOUTUBE SHORTS PROCESSING PIPELINE")
    print("=" * 60)
    
    API_KEY = "m08pEupw2UG4SaYq62oJoGlM_uHttAVD"
    API_SECRET = "xBY3hvMsEoCxMEmno6eeMrDKXZPe1mkd"
    
    youtube_urls = read_youtube_urls_from_file()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Direktori sementara: {temp_dir}")
        
        for youtube_url, video_id in youtube_urls:
            print(f"\n{'='*60}")
            print(f"Memproses URL: {youtube_url}")
            print(f"Video ID: {video_id}")
            print(f"{'='*60}")
            
            download_dir = os.path.join(temp_dir, video_id)
            os.makedirs(download_dir, exist_ok=True)
            
            print("Mengambil daftar shorts...")
            shorts_data = get_video_shorts(video_id)
            if not shorts_data:
                print("Tidak bisa mengambil data shorts")
                continue
            
            filtered_shorts = []
            if 'shorts' in shorts_data and isinstance(shorts_data['shorts'], list):
                for short in shorts_data['shorts']:
                    if ('viralityScore' in short and 
                        isinstance(short['viralityScore'], (int, float)) and 
                        short['viralityScore'] >= 70):
                        filtered_shorts.append(short)
            
            if not filtered_shorts:
                print("Tidak ada shorts dengan viralityScore ≥ 70")
                continue
            
            print(f"Ditemukan {len(filtered_shorts)} shorts (viralityScore ≥ 70)")
            short_ids = [short.get('_id') for short in filtered_shorts if short.get('_id')]
            
            if short_ids:
                print(f"Mendownload {len(short_ids)} video...")
                max_workers = min(10, len(short_ids))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_id = {executor.submit(download_short_video, short_id, download_dir): short_id for short_id in short_ids}
                    for future in concurrent.futures.as_completed(future_to_id):
                        short_id = future_to_id[future]
                        try:
                            success = future.result()
                            if success:
                                print(f"{short_id}.mp4 berhasil didownload")
                            else:
                                print(f"Gagal mendownload {short_id}.mp4 setelah beberapa percobaan")
                        except Exception as e:
                            print(f"Exception pada download {short_id}: {e}")
            
            print("Memproses data shorts...")
            final_result = {
                'video_id': video_id,
                'youtube_url': youtube_url,
                'total_shorts': len(filtered_shorts),
                'shorts': []
            }
            
            max_workers = min(10, len(filtered_shorts))
            short_data_map = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_short = {executor.submit(process_single_short, short): short for short in filtered_shorts}
                for future in concurrent.futures.as_completed(future_to_short):
                    try:
                        short_result = future.result()
                        if short_result:
                            final_result['shorts'].append(short_result)
                            short_data_map[short_result['id']] = short_result
                            print(f"Data short {short_result['id']} berhasil diproses")
                        else:
                            print(f"Gagal memproses data short")
                    except Exception as e:
                        print(f"Exception pada processing: {e}")
            
            output_filename = f'hasil_{video_id}.json'
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, indent=2, ensure_ascii=False)
                print(f"Data disimpan ke {output_filename}")
                print(f"Total shorts sukses: {len(final_result['shorts'])}/{len(filtered_shorts)}")
            except Exception as e:
                print(f"Gagal menyimpan ke file: {e}")
            
            print("\n" + "=" * 60)
            print("PROSES ANALISIS WAJAH DAN KONVERSI PORTRAIT")
            print("=" * 60)
            
            analyzer = VideoFaceAnalyzer(API_KEY, API_SECRET, temp_dir)
            
            for short_info in final_result['shorts']:
                short_id = short_info['id']
                title = short_info['title']
                input_video = os.path.join(download_dir, f"{short_id}.mp4")
                if not os.path.exists(input_video):
                    print(f"File video {input_video} tidak ditemukan, dilewati")
                    continue
                
                print(f"\nMemproses: {title}")
                print(f"   ID: {short_id}")
                
                print(f"   Analisis wajah...")
                scene_data = analyzer.analyze_video(
                    input_video=input_video,
                    scene_threshold=0.4
                )
                
                sanitized_title = sanitize_filename(title)
                output_video = f"{sanitized_title}.mp4"
                print(f"   Konversi ke portrait...")
                success = convert_to_portrait(input_video, scene_data, output_video, temp_dir)
                if success:
                    print(f"   Video final: {output_video}")
                else:
                    print(f"   Gagal mengonversi {short_id}")
        
        print("\n" + "=" * 60)
        print("PIPELINE SELESAI!")
        print("=" * 60)
        print("\nVideo-video hasil konversi telah disimpan dengan nama sesuai judul masing-masing.")
        print("Metadata lengkap tersimpan di file hasil_<video_id>.json")

if __name__ == "__main__":
    main_pipeline()
