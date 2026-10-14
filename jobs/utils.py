import hashlib
import io
import os
import tempfile

import pyarrow.fs as pafs

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get("TABULAR_DATA_PATH", os.path.join(THIS_DIR, "data", "products.parquet"))
FACE_MODEL = os.environ.get("FACE_MODEL_PATH", os.path.join(THIS_DIR, "models", "yolov11s-face.pt"))


FACE_CONF = 0.1
FACE_IMGSZ = 1280
BOX_PAD = 0.25
HOLD_PAST = 3
HOLD_FUTURE = 12
DRIFT = 0.25

DEFAULT_OSS_ENDPOINT = "https://s3.oss-cn-beijing-internal.aliyuncs.com"


def load_face_net():
    from ultralytics import YOLO

    return YOLO(FACE_MODEL)


def _grow(box, ratio, img_w, img_h):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * ratio, (y1 - y0) * ratio
    return max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(img_w, int(x1 + dx)), min(img_h, int(y1 + dy))


def detect_faces(model, bgr):
    result = model.predict(bgr, conf=FACE_CONF, imgsz=FACE_IMGSZ, device=0, verbose=False)[0]
    img_h, img_w = bgr.shape[:2]
    boxes = [_grow(b, BOX_PAD, img_w, img_h) for b in result.boxes.xyxy.tolist()]
    return [b for b in boxes if b[2] > b[0] and b[3] > b[1]]


def mosaic(img, boxes, cells=6, min_block=10):
    import cv2

    for x0, y0, x1, y1 in boxes:
        block = max(min_block, min(x1 - x0, y1 - y0) // cells)
        small = cv2.resize(img[y0:y1, x0:x1], (max(1, (x1 - x0) // block), max(1, (y1 - y0) // block)),
                           interpolation=cv2.INTER_AREA)
        img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return img


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _borrow(win, i):
    img_h, img_w = win[i][2].shape[:2]
    own = win[i][3]
    boxes = list(own)
    future = range(i + 1, min(len(win), i + HOLD_FUTURE + 1))
    past = range(i - 1, max(-1, i - HOLD_PAST - 1), -1)
    for js in (future, past):
        seen = list(own)
        for j in js:
            boxes += [_grow(b, DRIFT * abs(i - j), img_w, img_h)
                      for b in win[j][3] if not any(_overlap(b, s) for s in seen)]
            seen += win[j][3]
    return boxes


def strip_scheme(uri):
    return uri.removeprefix("s3://")


def oss_endpoint():
    return os.environ.get("AWS_ENDPOINT_URL", DEFAULT_OSS_ENDPOINT)


def oss_filesystem():
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    return pafs.S3FileSystem(
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_override=oss_endpoint(),
        force_virtual_addressing=True,
    )


def video_id(path):
    return hashlib.md5(strip_scheme(path).encode()).hexdigest()[:16]


def mask_video(model, data, video_path, output_uri):
    import av

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        num_frames = 0
        win = []

        def write(i):
            pts, time_base, img, _ = win[i]
            out = av.VideoFrame.from_ndarray(mosaic(img, _borrow(win, i)), format="bgr24")
            out.pts, out.time_base = pts, time_base
            dst.mux(vout.encode(out))

        with av.open(io.BytesIO(data)) as src, av.open(tmp, "w") as dst:
            vin = src.streams.video[0]
            vout = dst.add_stream("libx264", rate=vin.average_rate)
            vout.width, vout.height, vout.pix_fmt = vin.codec_context.width, vin.codec_context.height, "yuv420p"
            vout.codec_context.time_base = vin.time_base
            for frame in src.decode(vin):
                bgr = frame.to_ndarray(format="bgr24")
                win.append((frame.pts, frame.time_base, bgr, detect_faces(model, bgr)))
                num_frames += 1
                if len(win) > HOLD_FUTURE:
                    write(len(win) - HOLD_FUTURE - 1)
                if len(win) > HOLD_PAST + HOLD_FUTURE:
                    win.pop(0)
            for i in range(max(0, len(win) - HOLD_FUTURE), len(win)):
                write(i)
            dst.mux(vout.encode())
        if not num_frames:
            raise RuntimeError(f"无法解码视频: {video_path}")
        dest = f"{strip_scheme(output_uri).rstrip('/')}/{video_id(video_path)}.mp4"
        pafs.copy_files(tmp, dest, source_filesystem=pafs.LocalFileSystem(),
                        destination_filesystem=oss_filesystem())
    finally:
        os.remove(tmp)
    return f"s3://{dest}", num_frames
