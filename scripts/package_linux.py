#!/usr/bin/env python3
"""Bundle the Linux executable and its ELF dependency closure, without host glibc/driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_LIBS = {
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "libresolv.so.2", "libutil.so.1", "ld-linux-x86-64.so.2",
    "libcuda.so.1", "libnvidia-ml.so.1",
}


def run(*args, env=None):
    return subprocess.check_output([str(x) for x in args], env=env, text=True, stderr=subprocess.STDOUT)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def dependencies(path, env):
    text = run("ldd", path, env=env)
    if "not found" in text:
        raise RuntimeError(f"Unresolved dependencies of {path}:\n{text}")
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/.*?)\s+\(0x", line)
        if match and match[1] not in SYSTEM_LIBS:
            source = Path(match[2]).resolve()
            if "stubs" in source.parts:
                raise RuntimeError(f"CUDA driver stub must not be shipped: {source}")
            yield match[1], source


def glibc_requirement(path):
    versions = re.findall(r"Name: GLIBC_(\d+(?:\.\d+)+)", run("readelf", "--version-info", path))
    return max((tuple(map(int, value.split("."))) for value in versions), default=(0, 0))


def licenses(prefixes, sources, destination):
    records = []
    for prefix in prefixes:
        relative = {str(p.relative_to(prefix)) for p in sources if p.is_relative_to(prefix)}
        for metadata in sorted((prefix / "conda-meta").glob("*.json")):
            item = json.loads(metadata.read_text())
            # Also retain notices for statically linked/header-only build dependencies.
            static = item["name"] in {"cryptopp", "cpp-expected", "eigen", "nlohmann_json", "spdlog", "fmt"}
            if not static and not relative.intersection(item.get("files", [])):
                continue
            key = f'{item["name"]}-{item["version"]}-{item["build"]}'
            cached = Path(item.get("link", {}).get("source", "")) / "info" / "licenses"
            copied = []
            if cached.is_dir():
                for source in sorted(cached.rglob("*")):
                    if source.is_file():
                        target = destination / key / source.relative_to(cached)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                        copied.append(str(target.relative_to(destination)))
            for relative_file in item.get("files", []):
                if "/licenses/" in relative_file:
                    source = prefix / relative_file
                    if source.is_file():
                        target = destination / key / Path(relative_file).name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                        copied.append(str(target.relative_to(destination)))
            gpl = Path("/usr/share/common-licenses/GPL-3")
            if "GPL-3.0" in item.get("license", "") and gpl.is_file():
                target = destination / key / "GPL-3.0.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(gpl, target)
                copied.append(str(target.relative_to(destination)))
            records.append({"package": key, "license": item.get("license"), "url": item.get("url"), "files": copied})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "packages.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/linux-cu128")
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/ai3d-waterpost-linux-x86_64-cu128")
    args = parser.parse_args()
    prefix, cuda, build, output = (p.resolve() for p in (args.prefix, args.cuda_root, args.build_dir, args.output))
    archive = output.with_name(output.name + ".tar.gz")
    if output.exists() or archive.exists():
        raise RuntimeError("Output already exists; select a new --output directory.")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:{cuda}/targets/x86_64-linux/lib:{cuda}/lib"
    programs = {name: build / "bin" / name for name in (
        "ai3d_waterpost", "ai3d_waterpost_tests", "ai3d_waterpost_gpu_smoke")}
    libraries = {}

    def collect(path):
        for name, source in dependencies(path, env):
            if name in libraries and libraries[name] != source:
                if digest(libraries[name]) != digest(source):
                    raise RuntimeError(f"Conflicting library {name}: {libraries[name]} and {source}")
                if not source.is_relative_to(prefix):
                    continue
            libraries[name] = source

    for path in programs.values():
        collect(path)
    # oneTBB loads its allocator/affinity helpers dynamically rather than through DT_NEEDED.
    for pattern in ("libtbbmalloc.so.*", "libtbbmalloc_proxy.so.*", "libtbbbind*.so.*"):
        for path in sorted((prefix / "lib").glob(pattern)):
            text = run("readelf", "-d", path)
            soname = re.search(r"\(SONAME\).*\[(.*?)\]", text)
            if soname:
                libraries[soname[1]] = path.resolve()
                collect(path)
    # ldd can omit aliases that resolve to one inode (e.g. BLAS and LAPACK).
    # Preserve every DT_NEEDED name, including those aliases.
    pending = list(programs.values()) + list(libraries.values())
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", run("readelf", "-d", path))
        for name in needed:
            if name in SYSTEM_LIBS:
                continue
            if name not in libraries:
                locations = [prefix / "lib" / name, cuda / "lib" / name,
                             cuda / "targets/x86_64-linux/lib" / name]
                source = next((p.resolve() for p in locations if p.is_file()), None)
                if source is None:
                    raise RuntimeError(f"Cannot resolve {name}, required by {path}")
                libraries[name] = source
                collect(source)
            pending.append(libraries[name])
    requirements = {name: glibc_requirement(path) for name, path in {**programs, **libraries}.items()}
    for name, version in requirements.items():
        if version > (2, 31):
            raise RuntimeError(f"{name} needs glibc {version}; incompatible with Ubuntu 20.04")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".waterpost-package-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        (stage / "bin").mkdir(parents=True)
        (stage / "lib").mkdir()
        copied_sources = {}
        for name, source in libraries.items():
            target = stage / "lib" / name
            if source in copied_sources:
                os.link(copied_sources[source], target)
            else:
                shutil.copyfile(source, target)
                subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(target)], check=True)
                copied_sources[source] = target
        for name, source in programs.items():
            target = stage / "bin" / name
            shutil.copyfile(source, target)
            target.chmod(0o755)
            subprocess.run(["patchelf", "--set-rpath", "$ORIGIN/../lib", str(target)], check=True)
        launcher = stage / "ai3d_waterpost"
        launcher.write_text('#!/bin/sh\nset -eu\nroot=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
                            'export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
                            'exec "$root/bin/ai3d_waterpost" "$@"\n')
        launcher.chmod(0o755)
        shutil.copyfile(ROOT / "scripts/linux-runtime-README.md", stage / "README.md")
        shutil.copyfile(ROOT / "README.md", stage / "CLI-README.md")
        licenses([prefix, cuda], set(libraries.values()), stage / "licenses")
        cache = (build / "CMakeCache.txt").read_text()
        archs = re.search(r"^AI3D_CUDA_ARCHITECTURES:[^=]+=(.*)$", cache, re.M)[1]
        info = {
            "platform": "linux-x86_64", "cuda": run(cuda / "bin/nvcc", "--version").strip(),
            "cuda_architectures": archs.split(";"),
            "maximum_required_glibc": ".".join(map(str, max(requirements.values()))),
            "glibc_requirements": {k: ".".join(map(str, v)) for k, v in requirements.items()},
            "git_revision": run("git", "-C", ROOT, "rev-parse", "HEAD").strip(),
            "git_status": run("git", "-c", "core.autocrlf=true", "-c", "core.safecrlf=false",
                              "-C", ROOT, "status", "--short", env={**env, "GIT_OPTIONAL_LOCKS": "0"}),
            "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                              for folder in ("src", "include", "apps", "tests")
                              for p in sorted((ROOT / folder).rglob("*")) if p.is_file()},
        }
        clean = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LANG": "C.UTF-8"}
        # All non-system dependencies must resolve within the relocated package.
        for path in [*map(lambda name: stage / "bin" / name, programs),
                     *map(lambda name: stage / "lib" / name, libraries)]:
            for name, resolved in dependencies(path, clean):
                if not resolved.is_relative_to(stage):
                    raise RuntimeError(f"Package still depends on an external library: {name} => {resolved}")
        run(stage / "bin/ai3d_waterpost", "--help", env=clean)
        validation = []
        for name in ("ai3d_waterpost_tests", "ai3d_waterpost_gpu_smoke"):
            result = run(stage / "bin" / name, env=clean)
            print(result, end="")
            validation.append(f"{name}\n{result}")
        (stage / "validation.txt").write_text("\n".join(validation))
        info["elf_sha256"] = {str(p.relative_to(stage)): digest(p)
                              for folder in ("bin", "lib") for p in sorted((stage / folder).iterdir())}
        (stage / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
        stage.rename(output)

    def normalize(member):
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        member.mode = 0o755 if member.isdir() or "/bin/" in member.name or member.name.endswith("/ai3d_waterpost") else 0o644
        return member

    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        tar.add(output, arcname=output.name, filter=normalize)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "size_mib": round(archive.stat().st_size / 2**20, 2),
                      "libraries": len(libraries), "required_glibc": info["maximum_required_glibc"]}))


if __name__ == "__main__":
    main()
