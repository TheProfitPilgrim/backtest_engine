import os
import json

# Function to extract the Python code from a .py or .ipynb file
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


# Function to create the Markdown content
def create_markdown_from_code(root_dir, output_md):
    with open(output_md, 'w', encoding='utf-8') as md_file:
        for root, dirs, files in os.walk(root_dir):

            # Skip .venv entirely
            if '.venv' in dirs:
                dirs.remove('.venv')

            for file in files:
                if file.endswith(('.py', '.ipynb')):  # Include .py and .ipynb
                    file_path = os.path.join(root, file)
                    code = extract_code_from_file(file_path)

                    if not code.strip():
                        continue  # Skip empty results (e.g., ipynb with no code cells)

                    # Write the title (file name) and the code to the markdown file
                    md_file.write(f"## {file}\n\n")
                    md_file.write("```python\n")
                    md_file.write(code)
                    md_file.write("\n```\n\n")


# Define the root directory and the output markdown file
root_directory = '.'  # Current directory (change this if needed)
output_markdown = 'all_code.md'

# Create the markdown file
create_markdown_from_code(root_directory, output_markdown)

print(f"Markdown file '{output_markdown}' has been created with all code.")