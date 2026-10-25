import json
from pathlib import Path


notebook_path = Path("notebooks/rag_experiments.ipynb")

with notebook_path.open(
    "r",
    encoding="utf-8"
) as file:
    notebook = json.load(file)


imports = []
functions = []


for cell in notebook["cells"]:

    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])

    for line in source.splitlines():

        stripped = line.strip()

        if stripped.startswith("import "):
            imports.append(stripped)

        elif stripped.startswith("from "):
            imports.append(stripped)

        elif stripped.startswith("def "):
            functions.append(stripped)


print("\n" + "=" * 80)
print("IMPORTS")
print("=" * 80)

for item in sorted(set(imports)):
    print(item)


print("\n" + "=" * 80)
print("FUNCTIONS")
print("=" * 80)

for item in functions:
    print(item)