from __future__ import annotations

import argparse

from collections import Counter

import ctypes

import hashlib

import json

from pathlib import Path

import re

import shutil

import subprocess

CODE = Path(__file__).resolve().parents[1]

SOURCE_SUFFIXES = {".cu", ".cpp", ".cc", ".cxx", ".c"}

HEADER_SUFFIXES = {".h", ".hpp", ".cuh"}

ABI_HEADERS = {"coil_field.h": "include/coil_field.h",
               "coil_field_internal.h": "src/coil_field_internal.h"}

OPTIONAL_TORSION = {"sgpu_coil_torsion_gradient", "sgpu_coil_torsion_p95_gradient"}

INCLUDE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)

LITERAL_OR_COMMENT = re.compile(
    r'//[^\n]*|/\*.*?\*/|(?:u8|u|U|L)?R"(?P<tag>[^ ()\\\t\r\n]{0,16})'
    r'\(.*?\)(?P=tag)"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)

ABI_SIGNATURE = (r'^[ \t]*(?:[A-Za-z_]\w*(?:::\w+)*(?:<[^>\n]+>)?[ \t*&]+)+'
                 r'(sgpu_\w+)\s*\([^;{}]*\)\s*')

ABI_DECLARATION = re.compile(ABI_SIGNATURE + r';', re.DOTALL | re.MULTILINE)

ABI_DEFINITION = re.compile(ABI_SIGNATURE + r'\{', re.DOTALL | re.MULTILINE)

THREAD_LOCAL = re.compile(
    r'(?P<extern>\bextern\s+)?\bthread_local\s+[^;{}=]*?\b(?P<name>[A-Za-z_]\w*)\s*(?P<ending>[;={])'
)

def normalized(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")

def masked(text: str) -> str:
    """Remove literals/comments but preserve positions, lines and C++ braces."""
    return LITERAL_OR_COMMENT.sub(
        lambda match: "".join("\n" if char == "\n" else " " for char in match[0]), text
    )

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def within(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), f"path escapes root: {relative}")
    return result

def cmake_sources(native: Path) -> set[Path]:
    # Explicit source paths are intentional: a newly added TU must be reviewed
    # in CMake instead of entering the build through an unrestricted file glob.
    text = re.sub(r"#[^\n]*", "", normalized(native / "CMakeLists.txt"))
    paths = re.findall(r'(?<![\w/])src/[\w./+-]+\.(?:cu|cpp|cc|cxx|c)\b', text)
    require(paths, "CMake has no explicit native implementation source paths")
    return {within(native, path) for path in paths}

def check_scopes(path: Path, text: str) -> None:
    cleaned = masked(text)
    depth = 0
    for number, line in enumerate(cleaned.splitlines(), 1):
        for token in line:
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
                require(depth >= 0, f"unmatched closing scope: {path}:{number}")
    require(depth == 0, f"file ends inside a scope (possible split function): {path}")
    # A complete file can still textually inject an implementation fragment at
    # a nested scope. Reject local includes inside functions/namespaces too.
    for include in INCLUDE.finditer(text):
        before = cleaned[:include.start()]
        nesting = before.count("{") - before.count("}")
        require(nesting == 0, f"include inside an open scope: {path}:{text.count(chr(10), 0, include.start())+1}")

def check_layout(native: Path) -> dict:
    sources = {path.resolve() for path in (native / "src").rglob("*")
               if path.is_file() and path.suffix in SOURCE_SUFFIXES}
    require(sources, "native/src contains no implementation translation units")
    listed = cmake_sources(native)
    require(sources == listed,
            "CMake source registration mismatch: " + json.dumps({
                "unlisted": sorted(str(p.relative_to(native)) for p in sources-listed),
                "missing": sorted(str(p.relative_to(native)) for p in listed-sources),
            }))
    files = sorted(path.resolve() for folder in (native / "src", native / "include")
                   for path in folder.rglob("*") if path.is_file()
                   and path.suffix in SOURCE_SUFFIXES | HEADER_SUFFIXES | {".inc"})
    require(not any(path.suffix == ".inc" for path in files),
            "native contains textual .inc implementation fragments")
    states: dict[str, list[str]] = {}
    include_edges = []
    definitions = Counter()
    for path in files:
        text = normalized(path)
        check_scopes(path, text)
        for include in INCLUDE.finditer(text):
            name = include[1]
            require(Path(name).suffix not in SOURCE_SUFFIXES | {".inc"},
                    f"implementation include is forbidden: {path}: {name}")
            # System includes are resolved by the compiler/toolkit, not here.
            directive = text[include.start():include.end()]
            if '"' not in directive:
                continue
            target = next((candidate.resolve() for candidate in
                           (path.parent/name, native/"include"/name)
                           if candidate.is_file()), None)
            require(target is not None and target.is_relative_to(native),
                    f"unresolved or nonlocal include: {path}: {name}")
            include_edges.append([path.relative_to(native).as_posix(), target.relative_to(native).as_posix()])
        clean = masked(text)
        for declaration in THREAD_LOCAL.finditer(clean):
            if declaration["extern"] and declaration["ending"] == ";":
                continue
            name = declaration["name"]
            require(path.suffix in SOURCE_SUFFIXES,
                    f"thread-local state defined in header: {path}: {name}")
            states.setdefault(name, []).append(path.relative_to(native).as_posix())
        if path in sources:
            definitions.update(ABI_DEFINITION.findall(clean))
    require(all(len(owners) == 1 for owners in states.values()),
            "duplicate thread-local state definitions: " + json.dumps({
                name: owners for name, owners in states.items() if len(owners) > 1}))
    require(all(count == 1 for count in definitions.values()),
            "duplicate sgpu_ API definitions: " + json.dumps(dict(definitions)))
    return {"translation_units": sorted(path.relative_to(native).as_posix() for path in sources),
            "thread_local_owners": states, "include_edges": include_edges,
            "api_definitions": sorted(definitions)}

if __name__ == "__main__":
    print(json.dumps(check_layout(CODE/"native"), indent=2))
