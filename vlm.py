"""VLM calls and crop-sheet verification for candidate room changes."""
import base64, hashlib, io, json, os, shutil, subprocess, tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

TILE, COLS, ROWS = 320, 4, 3
CONTEXT = 2.5

VERIFY_PROMPT = """These are two phone videos of the same room. Video A is before and video B is after.
The first images are labeled contact sheets of both videos. The remaining images are candidate crops; each tile
is labeled #<id> <video> f<frame>, and the candidate region is outlined in magenta. A candidate shown only in A
was detected as removed, only in B as added, and in both as moved.

For every candidate, decide whether the outlined physical object really was added, removed, or moved. Check the
contact sheets for that object in the other video and for its position relative to nearby furniture. Ignore camera
viewpoint, occlusion, lighting, shadows, people, and unchanged parts of walls, floors, and furniture.
Return one entry for every candidate id: {ids}. Give a short object name and one sentence describing its location
relative to nearby furniture. Confidence is how sure you are of your real/false verdict, from 0 to 1."""

VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["candidates"], "properties": {
    "candidates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                   "required": ["id", "real", "name", "description", "confidence"], "properties": {
        "id": {"type": "integer"}, "real": {"type": "boolean"}, "name": {"type": "string"},
        "description": {"type": "string"}, "confidence": {"type": "number"}}}}}}


def backend() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if shutil.which("codex"):
        return "codex"
    return None


def _jpeg(im):
    out = io.BytesIO()
    im.convert("RGB").save(out, "JPEG", quality=90)
    return out.getvalue()


def _validate(value, schema):
    kind = schema.get("type")
    types = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
             "string": lambda v: isinstance(v, str), "integer": lambda v: type(v) is int,
             "number": lambda v: type(v) in (int, float), "boolean": lambda v: type(v) is bool,
             "null": lambda v: v is None}
    if kind and (kind not in types or not types[kind](value)):
        raise ValueError(f"VLM response expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("VLM response has a value outside the schema enum")
    if kind == "object":
        props = schema.get("properties", {})
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"VLM response missing {sorted(missing)}")
        if schema.get("additionalProperties") is False and value.keys() - props.keys():
            raise ValueError("VLM response has unexpected fields")
        for key in value.keys() & props.keys():
            _validate(value[key], props[key])
    if kind == "array" and "items" in schema:
        for item in value:
            _validate(item, schema["items"])
    if kind == "number" and not np.isfinite(value):
        raise ValueError("VLM response has a non-finite number")


def _parse_json(raw, schema):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
        _validate(value, schema)
        return value
    except (json.JSONDecodeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            _validate(value, schema)
            return value
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"VLM response did not contain JSON matching the schema: {text[:500]}")


def ask(prompt: str, images: list[Image.Image], schema: dict, name: str) -> dict:
    """Make one VLM call and return a schema-checked JSON object."""
    selected = backend()
    if selected == "openai":
        from openai import OpenAI
        content = [{"type": "text", "text": prompt}]
        for im in images:
            url = "data:image/jpeg;base64," + base64.b64encode(_jpeg(im)).decode()
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        result = OpenAI().chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-5"), messages=[{"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}})
        return _parse_json(result.choices[0].message.content, schema)
    if selected == "codex":
        with tempfile.TemporaryDirectory(prefix="scenediff-vlm-") as td:
            tmp = Path(td)
            command = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", prompt]
            help_text = subprocess.run(["codex", "exec", "--help"], capture_output=True, text=True, check=True).stdout
            if "--output-schema" in help_text:
                schema_path = tmp / "schema.json"
                schema_path.write_text(json.dumps(schema))
                command += ["--output-schema", str(schema_path)]
            else:
                command[-1] += "\nReturn JSON only, without markdown, matching this schema: " + json.dumps(schema)
            for i, im in enumerate(images):
                path = tmp / f"image_{i:03d}.jpg"
                path.write_bytes(_jpeg(im))
                command += ["-i", str(path)]
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"codex VLM call failed: {result.stderr[-1500:]}")
        return _parse_json(result.stdout, schema)
    raise RuntimeError("No VLM backend: set OPENAI_API_KEY, install codex, or use the offline version.")


def tile(frame, mask, label, font=None):
    """A square crop around a mask with a magenta outline and numbered label."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError("Candidate mask is empty")
    height, width = mask.shape
    side = min(min(height, width), max(int(CONTEXT * max(np.ptp(ys), np.ptp(xs))), min(height, width) // 2))
    cy, cx = (int(ys.min()) + int(ys.max())) // 2, (int(xs.min()) + int(xs.max())) // 2
    y0 = int(np.clip(cy - side // 2, 0, height - side))
    x0 = int(np.clip(cx - side // 2, 0, width - side))
    im = np.asarray(Image.open(frame).convert("RGB")).copy()
    if im.shape[:2] != mask.shape:
        raise ValueError("Candidate mask and frame have different sizes")
    edge = mask & ~ndimage.binary_erosion(mask, iterations=3)
    im[edge] = (255, 0, 255)
    crop = Image.fromarray(im[y0:y0 + side, x0:x0 + side]).resize((TILE, TILE - 22), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (TILE, TILE), "#202124")
    out.paste(crop, (0, 22))
    ImageDraw.Draw(out).text((5, 3), label, fill="white", font=font)
    return out


def candidate_sheets(cands, frames):
    try:
        font = ImageFont.truetype("Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for i, masks in enumerate(cands):
        for video in "AB":
            views = masks.get(video, {})
            if views:
                frame = max(views, key=lambda f: np.asarray(views[f]).sum())
                tiles.append(tile(frames[video][frame], views[frame], f"#{i} {video} f{frame}", font))
    out = []
    for start in range(0, len(tiles), COLS * ROWS):
        sheet = Image.new("RGB", (COLS * TILE, ROWS * TILE), "#202124")
        for j, im in enumerate(tiles[start:start + COLS * ROWS]):
            sheet.paste(im, ((j % COLS) * TILE, (j // COLS) * TILE))
        out.append(sheet)
    return out


def _cache_key(cands, frames):
    h = hashlib.sha256()
    h.update(b"vlm-verify-v1")
    h.update(str(len(cands)).encode())
    for video in "AB":
        for path in frames[video]:
            h.update(json.dumps([video, str(path)]).encode())
    for i, masks in enumerate(cands):
        for video in "AB":
            for frame, mask in sorted(masks.get(video, {}).items()):
                arr = np.ascontiguousarray(mask, dtype=bool)
                h.update(json.dumps([i, video, frame, arr.shape]).encode())
                h.update(arr.tobytes())
    return h.hexdigest()


def verify(cands, frames, cache: Path) -> dict[int, dict]:
    """Check numbered mask candidates against both videos' contact sheets."""
    if not cands:
        return {}
    cache = Path(cache)
    key = _cache_key(cands, frames)
    try:
        saved = json.loads(cache.read_text())
        if saved["key"] == key:
            return {int(i): answer for i, answer in saved["answers"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    from semantic import sheets
    images = sheets(frames["A"], "A") + sheets(frames["B"], "B") + candidate_sheets(cands, frames)
    prompt = VERIFY_PROMPT.format(ids=", ".join(map(str, range(len(cands)))))
    response = ask(prompt, images, VERIFY_SCHEMA, "verify_candidates")
    _validate(response, VERIFY_SCHEMA)
    answers = {}
    for row in response["candidates"]:
        i, confidence = row["id"], float(np.clip(row["confidence"], 0, 1))
        if 0 <= i < len(cands) and i not in answers:
            answers[i] = {"real": row["real"], "name": row["name"], "description": row["description"],
                          "p_real": confidence if row["real"] else 1 - confidence}
    for i in range(len(cands)):  # an id the model skipped stays in, undecided
        answers.setdefault(i, {"real": True, "name": "object", "description": "", "p_real": 0.5})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "answers": answers}, indent=1))
    return answers
