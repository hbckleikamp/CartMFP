# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:30:59 2026

@author: e_kle
"""

# generate_dependencies_and_data_toml.py
import ast
import sys
from pathlib import Path
import pkg_resources
import os

# -----------------------------
# 1. Collect all imports from script(s)
# -----------------------------
def get_imports_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        node = ast.parse(f.read(), filename=file_path)
    imports = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.add(n.module.split(".")[0])
    return imports

def collect_imports(scripts):
    all_imports = set()
    for script in scripts:
        all_imports.update(get_imports_from_file(script))
    return all_imports

# -----------------------------
# 2. Filter to installed packages
# -----------------------------
def get_installed_packages():
    return {d.project_name.lower(): d.version for d in pkg_resources.working_set}

def filter_third_party(imports, installed_packages):
    deps = []
    for imp in imports:
        if imp.lower() in installed_packages:
            deps.append(f"{imp}=={installed_packages[imp.lower()]}")
    return sorted(deps)

# -----------------------------
# 3. Collect data files in the same folder as scripts
# -----------------------------
def collect_data_files(scripts):
    files_set = set()
    for script in scripts:
        folder = Path(script).parent
        for f in folder.iterdir():
            if f.is_file() and not f.name.endswith(".py") and not f.name.startswith("__"):
                # include non-Python files (like CSV)
                files_set.add(f.name)
    return sorted(files_set)

# -----------------------------
# 4. Main
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dependencies_and_data_toml.py <script1.py> [script2.py ...]")
        sys.exit(1)

    scripts = sys.argv[1:]
    imports = collect_imports(scripts)
    installed_packages = get_installed_packages()
    deps = filter_third_party(imports, installed_packages)

    # Output TOML-ready dependencies
    print("# -------- dependencies --------")
    print("dependencies = [")
    for d in deps:
        print(f'    "{d}",')
    print("]")

    # Detect data files
    data_files = collect_data_files(scripts)
    if data_files:
        print("\n# -------- package data --------")
        print("[tool.setuptools.package-data]")
        print('"*" = [')
        for f in data_files:
            print(f'    "{f}",')
        print("]")