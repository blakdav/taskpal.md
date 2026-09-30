"""
Markdown-backed to-do list with projects and per-project archives.

The file on disk is the source of truth.

    ## Groceries            <- a project
    - [ ] oat milk <!--id:a1b2c3-->

    ### Archive             <- that project's own history
    - [x] eggs ✅ 2026-09-28 <!--id:d4e5f6-->

Every project keeps its own archive, so a grocery list you tick through
weekly builds a browsable record of what you've bought without burying
everything else. Each request does a read-modify-write, so Obsidian / vim /
Syncthing can touch the same file without this app losing their changes.
"""

import os
import re
import secrets
import threading
from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for

app = Flask(__name__)

TASKPAL_PATH = Path(os.environ.get("TASKPAL_PATH", "data/taskpal.md"))

# Where voice tasks land, and the home for anything written before the
# first `##` header.
INBOX = os.environ.get("TASKPAL_INBOX", "Inbox")

# The `###` subsection name, used inside every project.
ARCHIVE = os.environ.get("TASKPAL_ARCHIVE", "Archive")

# --- transcription -----------------------------------------------------
# Defaults to OpenAI. Point TRANSCRIBE_URL at a local speaches / LocalAI
# instance to move this off the cloud -- the request shape is identical.
TRANSCRIBE_URL = os.environ.get(
    "TRANSCRIBE_URL", "https://api.openai.com/v1/audio/transcriptions"
)
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
TRANSCRIBE_KEY = os.environ.get("OPENAI_API_KEY", "")
TRANSCRIBE_HINT = os.environ.get("TRANSCRIBE_HINT", "")

# Optional shared secret. When set, write routes require
# `Authorization: Bearer <token>`. Leave unset behind WireGuard.
TASKPAL_TOKEN = os.environ.get("TASKPAL_TOKEN", "")

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # OpenAI's per-request ceiling

_lock = threading.Lock()

# - [ ] Some task <!--id:a3f2c1-->
TASK_RE = re.compile(
    r"^(?P<indent>\s*)-\s\[(?P<mark>[ xX])\]\s*"
    r"(?P<text>.*?)"
    r"(?:\s*<!--id:(?P<id>[0-9a-f]{6})-->)?\s*$"
)

HEADER_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")       # project
SUB_RE = re.compile(r"^###\s+(?P<name>.+?)\s*$")         # subsection

# Completion date, Obsidian Tasks style: ✅ 2026-09-28
DONE_RE = re.compile(r"\s*✅\s*(?P<date>\d{4}-\d{2}-\d{2})")


def new_id() -> str:
    return secrets.token_hex(3)


def is_archive_head(line: str) -> bool:
    m = SUB_RE.match(line)
    return bool(m) and m.group("name").strip().lower() == ARCHIVE.lower()


# --- file io -----------------------------------------------------------

def ensure_file() -> None:
    TASKPAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TASKPAL_PATH.exists():
        TASKPAL_PATH.write_text(f"# To do\n\n## {INBOX}\n\n", encoding="utf-8")


def read_file() -> str:
    ensure_file()
    return TASKPAL_PATH.read_text(encoding="utf-8")


def write_file(content: str) -> None:
    """Atomic-ish write: temp file then rename, so a crash mid-write
    can't leave you with a truncated list."""
    ensure_file()
    tmp = TASKPAL_PATH.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(TASKPAL_PATH)


# --- parsing -----------------------------------------------------------

def parse(content: str):
    """Return (tasks, healed_content).

    Each task carries its project and whether it sits in that project's
    archive. Tasks missing an id get one assigned, so items added by hand in
    Obsidian become clickable here.
    """
    tasks, lines, changed = [], content.splitlines(), False
    current, archived = INBOX, False

    for i, line in enumerate(lines):
        if HEADER_RE.match(line):
            current = HEADER_RE.match(line).group("name")
            archived = False
            continue
        if SUB_RE.match(line):
            archived = is_archive_head(line)
            continue

        m = TASK_RE.match(line)
        if not m or not m.group("text").strip():
            continue

        tid = m.group("id")
        if not tid:
            tid = new_id()
            lines[i] = f"{m.group('indent')}- [{m.group('mark')}] " \
                       f"{m.group('text').strip()} <!--id:{tid}-->"
            changed = True

        body = m.group("text").strip()
        d = DONE_RE.search(body)
        if d:
            body = DONE_RE.sub("", body).strip()

        tasks.append({
            "id": tid,
            "text": body,
            "done": m.group("mark").lower() == "x",
            "done_on": d.group("date") if d else None,
            "project": current,
            "archived": archived,
            "raw": lines[i],
        })

    healed = "\n".join(lines) + "\n" if changed else content
    return tasks, healed


def project_list(content: str, tasks):
    """Ordered project names with open-task counts.

    Order follows the file, so rearranging headers in Obsidian rearranges the
    sidebar. Empty projects still appear -- a project you just made shouldn't
    vanish because you haven't filled it yet.
    """
    names = []
    for line in content.splitlines():
        h = HEADER_RE.match(line)
        if h and h.group("name") not in names:
            names.append(h.group("name"))

    for t in tasks:
        if t["project"] not in names:
            names.append(t["project"])

    # Inbox is where voice tasks land, so it stays pinned to the top.
    if INBOX in names:
        names.remove(INBOX)
    names.insert(0, INBOX)

    open_n, arch_n = {n: 0 for n in names}, {n: 0 for n in names}
    for t in tasks:
        if t["archived"]:
            arch_n[t["project"]] = arch_n.get(t["project"], 0) + 1
        elif not t["done"]:
            open_n[t["project"]] = open_n.get(t["project"], 0) + 1

    return [{"name": n, "open": open_n.get(n, 0), "archived": arch_n.get(n, 0)}
            for n in names]


# --- section geometry --------------------------------------------------

def split_sections(lines):
    """(preamble, [(name, body_lines), ...]) split on `##` only.

    A project's `### Archive` stays inside its body, so moving a project
    carries its history with it.
    """
    preamble, sections = [], []
    for line in lines:
        h = HEADER_RE.match(line)
        if h:
            sections.append([h.group("name"), []])
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    return preamble, sections


def join_sections(preamble, sections):
    """Rebuild the file, normalising to one blank line between sections."""
    out = list(preamble)
    while out and not out[-1].strip():
        out.pop()

    for name, body in sections:
        trimmed = list(body)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        if out:
            out.append("")
        out.append(f"## {name}")
        out.extend(trimmed)

    return "\n".join(out) + "\n"


def section_bounds(lines, project: str, archived: bool = False):
    """(start, end) for where tasks live in a project.

    With archived=False this is the live area, which stops at the project's
    `### Archive` header -- so a new task never lands in the history. With
    archived=True it's the archive subsection's body, or None if absent.
    """
    start = None
    for i, line in enumerate(lines):
        h = HEADER_RE.match(line)
        if h and h.group("name") == project:
            start = i + 1
            break
    if start is None:
        return None

    end = len(lines)
    arch_at = None
    for j in range(start, len(lines)):
        if HEADER_RE.match(lines[j]):
            end = j
            break
        if arch_at is None and is_archive_head(lines[j]):
            arch_at = j

    if archived:
        if arch_at is None:
            return None
        a_start, a_end = arch_at + 1, end
        while a_end > a_start and not lines[a_end - 1].strip():
            a_end -= 1
        return a_start, a_end

    live_end = arch_at if arch_at is not None else end
    while live_end > start and not lines[live_end - 1].strip():
        live_end -= 1
    return start, live_end


def insert_line(lines, project: str, line: str, archived: bool = False):
    """Put `line` at the end of a project's live or archive area, creating
    whatever's missing."""
    live = section_bounds(lines, project, archived=False)

    if live is None:                                  # project doesn't exist
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"## {project}")
        if archived:
            lines.append("")
            lines.append(f"### {ARCHIVE}")
        lines.append(line)
        return lines

    if not archived:
        lines.insert(live[1], line)
        return lines

    arch = section_bounds(lines, project, archived=True)
    if arch is None:                                  # no archive yet
        at = live[1]
        lines.insert(at, "")
        lines.insert(at + 1, f"### {ARCHIVE}")
        lines.insert(at + 2, line)
    else:
        lines.insert(arch[1], line)
    return lines


# --- healing -----------------------------------------------------------

def migrate_global_archive(content: str) -> str:
    """One-time: fold a legacy top-level `## Archive` into the Inbox's own
    archive, then drop the section. Earlier versions kept one global
    graveyard; those entries have no project of their own, so Inbox is where
    they land."""
    preamble, sections = split_sections(content.splitlines())
    idx = next((i for i, (n, _) in enumerate(sections)
                if n.strip().lower() == ARCHIVE.lower()), None)
    if idx is None:
        return content

    _, body = sections.pop(idx)
    entries = [ln for ln in body if TASK_RE.match(ln) and
               TASK_RE.match(ln).group("text").strip()]

    lines = join_sections(preamble, sections).splitlines()
    for entry in entries:
        insert_line(lines, INBOX, entry.strip(), archived=True)
    return "\n".join(lines) + "\n"


def hoist_inbox(content: str) -> str:
    """Move the Inbox section to the top of the file, matching the sidebar."""
    preamble, sections = split_sections(content.splitlines())
    idx = next((i for i, (n, _) in enumerate(sections) if n == INBOX), None)
    if idx in (None, 0):
        return content
    sections.insert(0, sections.pop(idx))
    return join_sections(preamble, sections)


def heal(content: str) -> str:
    return hoist_inbox(migrate_global_archive(content))


# --- views -------------------------------------------------------------

def render(active: str = None, archive: bool = False):
    with _lock:
        content = read_file()
        tasks, healed = parse(content)
        if healed != content:
            content = healed
        fixed = heal(content)
        if fixed != content:
            content = fixed
        if content != read_file():
            write_file(content)
            tasks, _ = parse(content)

    projects = project_list(content, tasks)
    known = {p["name"] for p in projects}
    if active is not None and active not in known:
        active = None

    if archive:
        shown = [t for t in tasks
                 if t["archived"] and (active is None or t["project"] == active)]
    elif active is None:
        shown = [t for t in tasks if not t["archived"]]
    else:
        shown = [t for t in tasks
                 if t["project"] == active and not t["archived"]]

    here = next((p for p in projects if p["name"] == active), None)

    return render_template(
        "index.html",
        open_tasks=[t for t in shown if not t["done"]],
        done_tasks=[t for t in shown if t["done"]],
        projects=projects,
        active=active,
        archive=archive,
        total_open=sum(p["open"] for p in projects),
        here_archived=here["archived"] if here else 0,
        total_archived=sum(p["archived"] for p in projects),
        inbox=INBOX,
        path=TASKPAL_PATH.name,
    )


ALL = "*"        # sentinel in return_to meaning "the unfiltered view"
ARCHIVED = "@"   # ... and "the archive view"


@app.route("/")
def index():
    """Opens on whichever project sits at the top of the file."""
    with _lock:
        content = read_file()
    tasks, _ = parse(content)
    projects = project_list(content, tasks)
    return render(projects[0]["name"] if projects else None)


@app.route("/all")
def all_view():
    return render(None)


@app.route("/p/<path:project>")
def project_view(project):
    return render(unquote(project))


@app.route("/archive")
def archive_view():
    return render(None, archive=True)


@app.route("/p/<path:project>/archive")
def project_archive(project):
    return render(unquote(project), archive=True)


@app.route("/state")
def state():
    """Cheap change check: the file's mtime. The page polls this and only
    pulls new markup when the number moves."""
    ensure_file()
    return jsonify(mtime=TASKPAL_PATH.stat().st_mtime)


def back_to(target: str = None):
    """`target` is a project name, or a sentinel. A project name may carry an
    `@` suffix meaning that project's archive."""
    if target == ARCHIVED:
        return redirect(url_for("archive_view"))
    if target == ALL:
        return redirect(url_for("all_view"))
    if target and target.endswith(ARCHIVED):
        name = target[:-1]
        return redirect(url_for("project_archive", project=quote(name)))
    if target:
        return redirect(url_for("project_view", project=quote(target)))
    return redirect(url_for("index"))


# --- write helpers -----------------------------------------------------

def append_task(text: str, project: str = None) -> str:
    project = project or INBOX
    tid = new_id()
    entry = f"- [ ] {text} <!--id:{tid}-->"
    with _lock:
        lines = read_file().splitlines()
        insert_line(lines, project, entry)
        write_file("\n".join(lines) + "\n")
    return tid


def find_task(lines, task_id):
    for i, line in enumerate(lines):
        m = TASK_RE.match(line)
        if m and m.group("id") == task_id:
            return i, m
    return None, None


def authorised() -> bool:
    if not TASKPAL_TOKEN:
        return True
    header = request.headers.get("Authorization", "")
    supplied = header[7:] if header.startswith("Bearer ") else ""
    return secrets.compare_digest(supplied, TASKPAL_TOKEN)


def deny():
    return jsonify(error="unauthorised"), 401


# --- write routes ------------------------------------------------------

@app.route("/task", methods=["POST"])
def add_task():
    """Typed input. Stored verbatim -- no parsing, no LLM, no surprises."""
    if not authorised():
        return deny()

    payload = request.get_json(silent=True) or {}
    text = (request.form.get("text") or payload.get("text") or "").strip()
    project = (request.form.get("project")
               or payload.get("project") or INBOX).strip()

    if not text:
        return back_to(request.form.get("return_to"))

    append_task(text, project)

    if request.is_json:
        return jsonify(ok=True, text=text, project=project)
    return back_to(request.form.get("return_to"))


@app.route("/task/voice", methods=["POST"])
def add_task_voice():
    """Audio in, task out. Always lands in the inbox -- the Action Button has
    no UI to pick a project with, and triage is cheaper than dictation
    friction.

    Send the recording as multipart form-data under `file`. Responds with the
    transcript so the caller can show you what actually landed.
    """
    if not authorised():
        return deny()

    if not TRANSCRIBE_KEY:
        return jsonify(error="No OPENAI_API_KEY configured"), 503

    audio = request.files.get("file")
    if not audio or not audio.filename:
        return jsonify(error="No audio uploaded under field 'file'"), 400

    blob = audio.read()
    if not blob:
        return jsonify(error="Empty recording"), 400
    if len(blob) > MAX_AUDIO_BYTES:
        return jsonify(error="Recording over the 25 MB limit"), 413

    data = {"model": TRANSCRIBE_MODEL}
    if TRANSCRIBE_HINT:
        data["prompt"] = TRANSCRIBE_HINT

    try:
        resp = requests.post(
            TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {TRANSCRIBE_KEY}"},
            files={"file": (audio.filename, blob,
                            audio.mimetype or "application/octet-stream")},
            data=data,
            timeout=60,
        )
    except requests.RequestException as exc:
        return jsonify(error=f"Transcription unreachable: {exc}"), 502

    if resp.status_code != 200:
        return jsonify(error=f"Transcription failed ({resp.status_code})",
                       detail=resp.text[:400]), 502

    text = (resp.json().get("text") or "").strip()
    # Speech-to-text ends most utterances with a full stop. A task isn't a
    # sentence, so drop it.
    text = text.rstrip(".").strip()

    if not text:
        return jsonify(error="Nothing recognised in that recording"), 422

    append_task(text, INBOX)
    return jsonify(ok=True, text=text, project=INBOX)


@app.route("/rename/<task_id>", methods=["POST"])
def rename(task_id):
    """Edit a task's text in place, keeping its id, state and position."""
    if not authorised():
        return deny()

    text = (request.form.get("text") or "").strip()
    if not text:
        return back_to(request.form.get("return_to"))

    with _lock:
        lines = read_file().splitlines()
        i, m = find_task(lines, task_id)
        if i is not None:
            # Preserve a completion stamp the user didn't type.
            old = m.group("text").strip()
            d = DONE_RE.search(old)
            stamp = f" ✅ {d.group('date')}" if d else ""
            lines[i] = f"{m.group('indent')}- [{m.group('mark')}] " \
                       f"{text}{stamp} <!--id:{task_id}-->"
            write_file("\n".join(lines) + "\n")

    return back_to(request.form.get("return_to"))


@app.route("/task/<task_id>/<direction>", methods=["POST"])
def reorder_task(task_id, direction):
    """Swap a task with its neighbour inside the same list."""
    if not authorised():
        return deny()

    step = -1 if direction == "up" else 1 if direction == "down" else 0
    if not step:
        return back_to(request.form.get("return_to"))

    with _lock:
        lines = read_file().splitlines()
        i, _ = find_task(lines, task_id)
        if i is None:
            return back_to(request.form.get("return_to"))

        # Only swap within the same contiguous run of tasks, so a task can't
        # jump a heading into another project.
        j = i + step
        while 0 <= j < len(lines):
            if HEADER_RE.match(lines[j]) or SUB_RE.match(lines[j]):
                j = None
                break
            if TASK_RE.match(lines[j]) and TASK_RE.match(lines[j]).group("text").strip():
                break
            j += step
        else:
            j = None

        if j is not None and 0 <= j < len(lines):
            lines[i], lines[j] = lines[j], lines[i]
            write_file("\n".join(lines) + "\n")

    return back_to(request.form.get("return_to"))


@app.route("/move/<task_id>", methods=["POST"])
def move(task_id):
    """Pull a task out of its section and drop it at the end of another."""
    if not authorised():
        return deny()

    target = (request.form.get("project") or "").strip()
    if not target:
        return back_to(request.form.get("return_to"))

    with _lock:
        lines = read_file().splitlines()
        i, _ = find_task(lines, task_id)
        if i is not None:
            entry = lines.pop(i)
            insert_line(lines, target, entry.strip())
            write_file("\n".join(lines) + "\n")

    return back_to(request.form.get("return_to"))


@app.route("/project", methods=["POST"])
def add_project():
    if not authorised():
        return deny()

    name = (request.form.get("name") or "").strip().lstrip("#").strip()
    if not name:
        return back_to(request.form.get("return_to"))

    with _lock:
        preamble, sections = split_sections(read_file().splitlines())
        if not any(n == name for n, _ in sections):
            sections.append([name, []])
            write_file(join_sections(preamble, sections))

    return back_to(name)


@app.route("/project/<path:project>/<direction>", methods=["POST"])
def reorder_project(project, direction):
    """Swap a whole `## section` with its neighbour -- header, tasks, and the
    project's archive all move together."""
    if not authorised():
        return deny()

    project = unquote(project)
    step = -1 if direction == "up" else 1 if direction == "down" else 0
    if not step or project == INBOX:
        return back_to(project)

    with _lock:
        preamble, sections = split_sections(read_file().splitlines())

        # Inbox is pinned at the top, so it's never a valid swap partner.
        movable = [i for i, (n, _) in enumerate(sections) if n != INBOX]
        pos = next((k for k, i in enumerate(movable)
                    if sections[i][0] == project), None)

        if pos is not None and 0 <= pos + step < len(movable):
            a, b = movable[pos], movable[pos + step]
            sections[a], sections[b] = sections[b], sections[a]
            write_file(join_sections(preamble, sections))

    return back_to(project)


@app.route("/toggle/<task_id>", methods=["POST"])
def toggle(task_id):
    if not authorised():
        return deny()

    with _lock:
        lines = read_file().splitlines()
        i, m = find_task(lines, task_id)
        if i is not None:
            done = m.group("mark").lower() == "x"
            mark = " " if done else "x"
            text = m.group("text").strip()
            # Un-completing something drops its completion date, which is no
            # longer true. Re-completing it gets a fresh one on next archive.
            if done:
                text = DONE_RE.sub("", text).strip()
            lines[i] = f"{m.group('indent')}- [{mark}] " \
                       f"{text} <!--id:{task_id}-->"
            write_file("\n".join(lines) + "\n")

    return back_to(request.form.get("return_to"))


@app.route("/delete/<task_id>", methods=["POST"])
def delete(task_id):
    if not authorised():
        return deny()

    with _lock:
        lines = read_file().splitlines()
        kept = [ln for ln in lines
                if not ((m := TASK_RE.match(ln)) and m.group("id") == task_id)]
        write_file("\n".join(kept) + "\n")

    return back_to(request.form.get("return_to"))


@app.route("/archive-done", methods=["POST"])
def archive_done():
    """Move completed tasks into their OWN project's archive, stamped with
    today's date. Scoped to the current view.

    Nothing is deleted -- a finished task is a record, and keeping each
    project's record separate means ticking through a grocery list doesn't
    bury the history of everything else.
    """
    if not authorised():
        return deny()

    scope = (request.form.get("return_to") or "").strip()
    if scope in (ALL, ARCHIVED):
        scope = ""
    scope = scope.rstrip(ARCHIVED)

    stamp = date.today().isoformat()

    with _lock:
        lines = read_file().splitlines()
        kept, moved, current, in_arch = [], [], INBOX, False

        for line in lines:
            if HEADER_RE.match(line):
                current = HEADER_RE.match(line).group("name")
                in_arch = False
                kept.append(line)
                continue
            if SUB_RE.match(line):
                in_arch = is_archive_head(line)
                kept.append(line)
                continue

            m = TASK_RE.match(line)
            done = m and m.group("mark").lower() == "x"
            in_scope = (not scope) or current == scope

            if done and in_scope and not in_arch:
                text = m.group("text").strip()
                if not DONE_RE.search(text):
                    text = f"{text} ✅ {stamp}"
                tid = m.group("id") or new_id()
                moved.append((current, f"- [x] {text} <!--id:{tid}-->"))
                continue

            kept.append(line)

        for project, entry in moved:
            insert_line(kept, project, entry, archived=True)

        if moved:
            write_file("\n".join(kept) + "\n")

    return back_to(request.form.get("return_to"))


@app.route("/edit", methods=["GET", "POST"])
def edit():
    """Raw file editor. The escape hatch for anything the UI can't express."""
    ensure_file()

    if request.method == "POST":
        if not authorised():
            return deny()
        seen_mtime = float(request.form.get("mtime", 0))
        with _lock:
            current_mtime = TASKPAL_PATH.stat().st_mtime
            # Someone else wrote to the file since this page was served.
            # Refuse rather than clobber their work.
            if abs(current_mtime - seen_mtime) > 0.001:
                return render_template(
                    "edit.html",
                    content=request.form.get("content", ""),
                    mtime=current_mtime,
                    conflict=read_file(),
                ), 409
            write_file(request.form.get("content", ""))
        return redirect(url_for("index"))

    return render_template(
        "edit.html",
        content=read_file(),
        mtime=TASKPAL_PATH.stat().st_mtime,
        conflict=None,
    )


if __name__ == "__main__":
    ensure_file()
    app.run(host="0.0.0.0", port=8080, debug=True)
