#!/usr/bin/env python3
"""Lossless Sim2Sim delivery with receiver-confirmed mesh/texture reuse.

Python 3.10+ standard library only. Never launches a simulator or edits source
evidence. Inventory and archive digests are fingerprints, not signatures.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


RESOURCE_SUFFIXES = {
    ".stl", ".obj", ".dae", ".ply", ".msh", ".mesh",
    ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".dds", ".ktx", ".ktx2",
}
MANIFEST = "transfer_manifest.json"
MAX_MANIFEST_BYTES = 32 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(value):
    require(isinstance(value, str) and value and "\\" not in value, "Invalid relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(p not in ("", ".", "..") for p in value.split("/")),
            f"Unsafe relative path: {value}")
    return path


def local_path(value):
    path = Path(value).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlink path: {path}")
    return Path(os.path.abspath(path))


def digest(path):
    h = hashlib.sha256()
    with local_path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def load_json(path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    path = local_path(path)
    with path.open("xb") as stream:
        stream.write(data)


def resource_paths(path):
    paths = load_json(path)
    require(isinstance(paths, list) and all(isinstance(p, str) for p in paths),
            "Resources must be a JSON array of relative file paths")
    require(len(paths) == len(set(paths)), "Duplicate resource path")
    for name in paths:
        require(safe_path(name).suffix.lower() in RESOURCE_SUFFIXES,
                f"Not a supported mesh/texture resource: {name}")
    return set(paths)


def checked_file(path, expected_hash, size):
    path = local_path(path)
    return path.is_file() and path.stat().st_size == size and digest(path) == expected_hash


def build_inventory(root, resources, cache, receiver_id, output):
    """Receiver-side only: copy and verify actual resources before confirming them."""
    root, cache, output = map(local_path, (root, cache, output))
    require(root.is_dir(), "Inventory root must be an existing directory")
    require(not cache.is_relative_to(root) and not output.is_relative_to(root),
            "Cache and inventory must be outside the sealed evidence root")
    require(not output.exists(), "Inventory output already exists")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", receiver_id), "Invalid receiver ID")
    cache.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in sorted(resource_paths(resources)):
        source = local_path(root / name)
        require(source.is_file(), f"Missing resource: {name}")
        sha, size = digest(source), source.stat().st_size
        target = local_path(cache / sha)
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix=".resource-", dir=cache) as tmp:
                staged = Path(tmp) / sha
                shutil.copyfile(source, staged)
                require(checked_file(staged, sha, size), f"Resource changed during copy: {name}")
                try:
                    os.link(staged, target)  # atomic publication, never overwrite
                except FileExistsError:
                    pass
        require(checked_file(target, sha, size), f"Corrupt cached resource: {sha}")
        entries[sha] = {"sha256": sha, "size_bytes": size}
    receipt = {"version": 1, "kind": "sim2sim-resource-inventory", "receiver_id": receiver_id,
               "resources": [entries[h] for h in sorted(entries)]}
    write_new(output, json_bytes(receipt))
    return {"inventory": str(output), "sha256": digest(output), "resources": len(entries)}


def inventory_binding(path, expected_hash, receiver_id):
    if path is None:
        require(expected_hash is None and receiver_id is None, "Inventory required with its hash/receiver ID")
        return None, {}
    require(valid_hash(expected_hash) and digest(path) == expected_hash, "Inventory SHA-256 mismatch")
    data = load_json(path)
    require(isinstance(data, dict) and data.get("version") == 1 and data.get("kind") == "sim2sim-resource-inventory",
            "Unsupported inventory")
    require(receiver_id and data.get("receiver_id") == receiver_id, "Inventory receiver ID mismatch")
    known = {}
    require(isinstance(data.get("resources"), list), "Invalid inventory resources")
    for item in data["resources"]:
        require(isinstance(item, dict), "Invalid inventory resource entry")
        sha, size = item["sha256"], item["size_bytes"]
        require(valid_hash(sha) and type(size) is int and size >= 0, "Invalid inventory resource")
        require(sha not in known, "Duplicate inventory hash")
        known[sha] = size
    return {"sha256": expected_hash, "receiver_id": receiver_id}, known


def pack(source, resources, output, inventory=None, inventory_sha256=None, receiver_id=None):
    source, output = map(local_path, (source, output))
    sidecar = local_path(str(output) + ".sha256")
    require(source.is_dir(), "Source must be a sealed evidence directory")
    require(not output.is_relative_to(source), "Archive must be outside the sealed source")
    require(not output.exists() and not sidecar.exists(), "Archive or digest already exists")
    selected = resource_paths(resources)
    binding, known = inventory_binding(inventory, inventory_sha256, receiver_id)
    files = []
    for path in sorted(source.rglob("*")):
        path = local_path(path)
        if path.is_dir():
            continue
        require(path.is_file(), f"Not a regular source file: {path}")
        name = path.relative_to(source).as_posix()
        safe_path(name)
        sha, size = digest(path), path.stat().st_size
        is_resource = name in selected
        files.append({"path": name, "sha256": sha, "size_bytes": size,
                      "mode": path.stat().st_mode & 0o777,
                      "resource": is_resource,
                      "storage": "cache" if is_resource and known.get(sha) == size else "bundled"})
    require(selected <= {f["path"] for f in files}, "Resource list contains absent files")
    require(files, "Empty evidence directory")
    manifest = {"version": 1, "kind": "sim2sim-transfer", "source_name": source.name,
                "inventory": binding, "files": files}
    metadata = json_bytes(manifest)
    require(len(metadata) <= MAX_MANIFEST_BYTES, "Transfer manifest too large")
    with tempfile.TemporaryDirectory(prefix=".sim2sim-pack-", dir=output.parent) as tmp:
        staged = Path(tmp) / "bundle.tar.gz"
        with tarfile.open(staged, "w:gz", compresslevel=6) as archive:
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(metadata)
            archive.addfile(info, io.BytesIO(metadata))
            for entry in files:
                if entry["storage"] != "bundled":
                    continue
                path = source / entry["path"]
                info = tarfile.TarInfo("payload/" + entry["path"])
                info.size, info.mode = entry["size_bytes"], entry["mode"]
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        # Verify the actual compressed payload, not just pre-copy source hashes.
        with tarfile.open(staged, "r:gz") as archive:
            for entry in files:
                require(checked_file(source / entry["path"], entry["sha256"], entry["size_bytes"]),
                        f"Source changed while packaging: {entry['path']}")
                if entry["storage"] == "bundled":
                    with archive.extractfile("payload/" + entry["path"]) as stream:
                        verify_stream(stream, entry)
        require({p.relative_to(source).as_posix() for p in source.rglob("*") if local_path(p).is_file()}
                == {f["path"] for f in files}, "Source file inventory changed while packaging")
        sha = digest(staged)
        os.link(staged, output)
        write_new(sidecar, f"{sha}  {output.name}\n".encode())
    return {"archive": str(output), "sha256": sha, "size_bytes": output.stat().st_size,
            "bundled_files": sum(f["storage"] == "bundled" for f in files),
            "cached_resources": sum(f["storage"] == "cache" for f in files)}


def verify_stream(stream, entry, target=None):
    h, size = hashlib.sha256(), 0
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        require(size <= entry["size_bytes"], f"Oversized payload: {entry['path']}")
        h.update(chunk)
        if target is not None:
            target.write(chunk)
    require(size == entry["size_bytes"] and h.hexdigest() == entry["sha256"],
            f"Payload SHA-256/size mismatch: {entry['path']}")


def read_transfer(archive):
    members = archive.getmembers()
    names = [m.name for m in members]
    require(len(names) == len(set(names)), "Duplicate archive member")
    require(all(m.isfile() for m in members), "Only regular archive members allowed")
    require(MANIFEST in names, "Missing transfer manifest")
    require(archive.getmember(MANIFEST).size <= MAX_MANIFEST_BYTES, "Transfer manifest too large")
    with archive.extractfile(MANIFEST) as stream:
        metadata = stream.read()
    data = json.loads(metadata)
    require(isinstance(data, dict) and data.get("version") == 1 and data.get("kind") == "sim2sim-transfer",
            "Unsupported transfer")
    require(isinstance(data.get("files"), list), "Invalid transfer file list")
    paths = set()
    expected = {MANIFEST}
    for entry in data["files"]:
        require(isinstance(entry, dict), "Invalid transfer file entry")
        name = entry["path"]
        safe_path(name)
        require(name not in paths, "Duplicate payload path")
        paths.add(name)
        require(valid_hash(entry["sha256"]) and type(entry["size_bytes"]) is int
                and entry["size_bytes"] >= 0, "Invalid file digest/size")
        require(type(entry["mode"]) is int and 0 <= entry["mode"] <= 0o777, "Invalid file mode")
        require(type(entry["resource"]) is bool, "Invalid resource flag")
        require(entry["storage"] in ("bundled", "cache"), "Invalid storage type")
        if entry["resource"]:
            require(PurePosixPath(name).suffix.lower() in RESOURCE_SUFFIXES, "Invalid resource type")
        if entry["storage"] == "cache":
            require(entry["resource"] and data.get("inventory"), "Only confirmed resources may use cache")
        else:
            member_name = "payload/" + name
            expected.add(member_name)
            require(member_name in names and archive.getmember(member_name).size == entry["size_bytes"],
                    f"Missing/wrong-sized archive payload: {name}")
    require(paths and set(names) == expected, "Unexpected or missing archive members")
    require(all(not any(str(p) in paths for p in PurePosixPath(name).parents if str(p) != ".")
                for name in paths), "File/directory path collision")
    if data.get("inventory"):
        require(isinstance(data["inventory"], dict) and valid_hash(data["inventory"].get("sha256"))
                and isinstance(data["inventory"].get("receiver_id"), str), "Invalid inventory binding")
    return data, metadata


class MissingResources(ValueError):
    def __init__(self, items):
        super().__init__("Required receiver resources are missing or corrupt")
        self.items = items


def receive(archive_path, expected_hash, output, cache, receiver_id=None):
    archive_path, output, cache = map(local_path, (archive_path, output, cache))
    require(not output.exists(), "Receive output already exists")
    require(valid_hash(expected_hash) and digest(archive_path) == expected_hash, "Archive SHA-256 mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        data, metadata = read_transfer(archive)
        binding = data.get("inventory")
        if binding:
            require(receiver_id == binding["receiver_id"], "Transfer receiver ID mismatch")
        missing = []
        for entry in data["files"]:
            if entry["storage"] == "cache":
                try:
                    valid = checked_file(cache / entry["sha256"], entry["sha256"], entry["size_bytes"])
                except ValueError:
                    valid = False
                if not valid:
                    missing.append({k: entry[k] for k in ("path", "sha256", "size_bytes")})
        if missing:
            raise MissingResources(missing)
        with tempfile.TemporaryDirectory(prefix=".sim2sim-receive-", dir=output.parent) as tmp:
            restored = Path(tmp) / "restored"
            restored.mkdir()
            for entry in data["files"]:
                destination = restored / "payload" / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = (local_path(cache / entry["sha256"]).open("rb") if entry["storage"] == "cache"
                          else archive.extractfile("payload/" + entry["path"]))
                with stream, destination.open("xb") as target:
                    verify_stream(stream, entry, target)
                destination.chmod(entry["mode"])
            write_new(restored / MANIFEST, metadata)
            require(not output.exists(), "Receive output appeared during validation")
            restored.rename(output)
    return {"status": "restored_byte_verified", "output": str(output), "files": len(data["files"]),
            "semantic_validation": "not_performed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Receiver: cache actual resources and attest hashes")
    for flag in ("root", "resources", "cache", "receiver-id", "output"):
        inventory.add_argument("--" + flag, required=True)
    bundle = commands.add_parser("pack", help="Sender: package sealed evidence without rewriting it")
    for flag in ("source", "resources", "output"):
        bundle.add_argument("--" + flag, required=True)
    for flag in ("inventory", "inventory-sha256", "receiver-id"):
        bundle.add_argument("--" + flag)
    restore = commands.add_parser("receive", help="Receiver: verify bytes and restore original relative paths")
    for flag in ("archive", "sha256", "output", "cache"):
        restore.add_argument("--" + flag, required=True)
    restore.add_argument("--receiver-id")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "inventory":
            result = build_inventory(**args)
        elif command == "pack":
            result = pack(**args)
        else:
            args["archive_path"] = args.pop("archive")
            args["expected_hash"] = args.pop("sha256")
            result = receive(**args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as exc:
        result = {"status": "failed", "error": str(exc)}
        if isinstance(exc, MissingResources):
            result["required_resources"] = exc.items
        print(json.dumps(result, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
