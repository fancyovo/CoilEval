"""Architecture guards independent of the number and ordering of native modules."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


CODE = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("verify_native_modules", CODE/"tools/check_layout.py")
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def minimal_native(tmp_path, extra_sources=()):
    native = tmp_path/"native"
    (native/"src").mkdir(parents=True)
    (native/"include").mkdir()
    (native/"src/main.cpp").write_text("int evaluate() { return 0; }\n", encoding="utf-8")
    sources = ("src/main.cpp", *extra_sources)
    (native/"CMakeLists.txt").write_text(
        "add_library(example SHARED\n" + "\n".join(sources) + "\n)\n", encoding="utf-8")
    return native


def test_actual_modular_native_sources():
    result = audit.check_layout(CODE/"native")
    assert result["translation_units"]
    assert result["thread_local_owners"]


def test_new_tu_requires_explicit_cmake_registration(tmp_path):
    native = minimal_native(tmp_path)
    (native/"src/missing.cpp").write_text("void new_stage() {}", encoding="utf-8")
    with pytest.raises(ValueError, match="CMake source registration mismatch"):
        audit.check_layout(native)


def test_including_an_implementation_is_rejected_even_when_both_tus_are_registered(tmp_path):
    native = minimal_native(tmp_path, ["src/other.cpp"])
    (native/"src/main.cpp").write_text('#include "other.cpp"\n', encoding="utf-8")
    (native/"src/other.cpp").write_text("int evaluate() { return 0; }", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation include is forbidden"):
        audit.check_layout(native)


def test_a_function_scope_cannot_continue_in_another_file():
    with pytest.raises(ValueError, match="ends inside a scope"):
        audit.check_scopes(Path("first.cu"), "int stage() {\n return 1;\n")
    # Text in comments and ordinary/raw literals must not invent C++ scopes.
    audit.check_scopes(Path("complete.cu"),
                       'int stage() { const char* a="{"; auto b=R"tag(})tag"; /* } */ return 1; }')


def test_a_header_cannot_inject_text_inside_an_open_function_scope():
    with pytest.raises(ValueError, match="include inside an open scope"):
        audit.check_scopes(Path("fragment.cu"), 'int stage() {\n#include "body.cuh"\n}\n')


@pytest.mark.parametrize("header_definition", ["thread_local int center_cache;\n",
                                              "extern thread_local int center_cache = 0;\n"])
def test_thread_local_cache_declarations_share_one_implementation_owner(tmp_path, header_definition):
    native = minimal_native(tmp_path)
    (native/"include/state.h").write_text("extern thread_local int center_cache;\n", encoding="utf-8")
    (native/"src/main.cpp").write_text('#include "state.h"\nthread_local int center_cache = 0;\n', encoding="utf-8")
    assert audit.check_layout(native)["thread_local_owners"] == {"center_cache": ["src/main.cpp"]}
    (native/"include/state.h").write_text(header_definition, encoding="utf-8")
    with pytest.raises(ValueError, match="thread-local state defined in header"):
        audit.check_layout(native)


def test_duplicate_cache_definitions_are_rejected(tmp_path):
    native = minimal_native(tmp_path, ["src/other.cpp"])
    for source in ("main.cpp", "other.cpp"):
        (native/"src"/source).write_text("thread_local int center_cache = 0;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate thread-local state definitions"):
        audit.check_layout(native)


def test_abi_definition_detection_ignores_nested_calls():
    text = """
    int sgpu_example(int count) { return count; }
    int caller() {
      if (sgpu_example(1)) { return 0; }
      return sgpu_example(2);
    }
    extern "C" const char* sgpu_error() { return "failed"; }
    """
    assert set(audit.ABI_DEFINITION.findall(audit.masked(text))) == {"sgpu_example", "sgpu_error"}
