"""把 COLMAP sparse（cameras.bin / images.bin）转成 ai3d_waterpost poses 模式用的位姿 JSON。

    python scripts/colmap_to_poses.py <sparse_dir> <out.json> [--every N]

输出格式（world -> camera：cam = R(qvec) * p + tvec，COLMAP 约定）：
    {"schema": "ai3d-waterpost.poses/1",
     "cameras": [{"id": 1, "model": "PINHOLE", "width": W, "height": H, "params": [fx, fy, cx, cy]}],
     "images":  [{"id": 1, "name": "a.jpg", "camera_id": 1, "qvec": [w, x, y, z], "tvec": [x, y, z]}]}
只依赖标准库。
"""
import argparse
import json
import struct

MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
          4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
          8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


def read_cameras(path):
    cameras = []
    with open(path, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            name, n = MODELS[model_id]
            params = list(struct.unpack("<%dd" % n, f.read(8 * n)))
            cameras.append({"id": camera_id, "model": name, "width": width, "height": height, "params": params})
    return cameras


def read_images(path):
    images = []
    with open(path, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack("<idddddddi", f.read(64))
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                name += c
            (points,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * points, 1)
            images.append({"id": image_id, "name": name.decode("utf-8"), "camera_id": camera_id,
                           "qvec": [qw, qx, qy, qz], "tvec": [tx, ty, tz]})
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sparse")
    parser.add_argument("output")
    parser.add_argument("--every", type=int, default=1, help="keep every N-th image (sorted by name)")
    args = parser.parse_args()
    cameras = read_cameras(args.sparse + "/cameras.bin")
    images = sorted(read_images(args.sparse + "/images.bin"), key=lambda item: item["name"])
    images = images[:: max(1, args.every)]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"schema": "ai3d-waterpost.poses/1", "cameras": cameras, "images": images}, f, ensure_ascii=False, indent=1)
    print("%d cameras, %d images -> %s" % (len(cameras), len(images), args.output))


if __name__ == "__main__":
    main()
