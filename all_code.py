import os
import json

# Function to extract Python code from .py or .ipynb
def extract_code_from_file(file_path):
    if file_path.endswith('.py'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    elif file_path.endswith('.ipynb'):
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        code_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                code = ''.join(cell.get('source', []))
                code_cells.append(code)

        return "\n\n".join(code_cells)

    return ""


# Function to generate a directory tree (ONLY .py and .ipynb, ignoring hidden folders)
def generate_tree(root_dir):
    tree_lines = []

    for root, dirs, files in os.walk(root_dir):

        # Remove hidden folders (anything starting with .)
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Folder entry
        level = root.replace(root_dir, "").count(os.sep)
        indent = "    " * level
        folder_name = os.path.basename(root) if root != root_dir else root_dir
        tree_lines.append(f"{indent}{folder_name}/")

        # Only include .py and .ipynb files (and skip hidden files)
        sub_indent = "    " * (level + 1)
        for f in files:
            if not f.startswith(".") and f.endswith((".py", ".ipynb")):
                tree_lines.append(f"{sub_indent}{f}")

    return "\n".join(tree_lines)


# Create Markdown file containing tree + extracted code
def create_markdown_from_code(root_dir, output_md):
    with open(output_md, 'w', encoding='utf-8') as md_file:

        # Write directory tree at the top
        tree_output = generate_tree(root_dir)
        md_file.write("# Directory Tree (Python & Notebook Files Only)\n\n")
        md_file.write("```\n")
        md_file.write(tree_output)
        md_file.write("\n```\n\n")

        # Extraction phase
        for root, dirs, files in os.walk(root_dir):

            # Remove hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if not file.startswith(".") and file.endswith((".py", ".ipynb")):
                    file_path = os.path.join(root, file)
                    code = extract_code_from_file(file_path)

                    if not code.strip():
                        continue

                    md_file.write(f"## {file}\n\n")
                    md_file.write("```python\n")
                    md_file.write(code)
                    md_file.write("\n```\n\n")


# Run
root_directory = "."
output_markdown = "all_code.md"
create_markdown_from_code(root_directory, output_markdown)

print(f"Markdown file '{output_markdown}' created.")