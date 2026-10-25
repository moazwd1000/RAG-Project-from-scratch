import ast
import json
from pathlib import Path


NOTEBOOK_PATH = Path(
    "notebooks/rag_experiments.ipynb"
)


PRODUCTION_FUNCTIONS = {
    "load_document",
    "cleaning",
    "chunk_documents_sentences",
    "create_embeddings",
    "compute_cos_similarity",
    "retrieve_top_k",
    "reranking",
    "context_from_indices",
}


with NOTEBOOK_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    notebook = json.load(file)


found_functions = {}


for cell in notebook["cells"]:

    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    for node in tree.body:

        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.name in PRODUCTION_FUNCTIONS
        ):

            function_source = ast.get_source_segment(
                source,
                node
            )

            if function_source:
                found_functions[node.name] = (
                    function_source
                )


for function_name in sorted(
    PRODUCTION_FUNCTIONS
):

    print("\n" + "=" * 80)
    print(function_name)
    print("=" * 80)

    if function_name in found_functions:
        print(found_functions[function_name])

    else:
        print("NOT FOUND")