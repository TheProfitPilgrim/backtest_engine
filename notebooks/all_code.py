import os

# Function to extract the Python code from a file
def extract_code_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to create the Markdown content
def create_markdown_from_code(root_dir, output_md):
    with open(output_md, 'w', encoding='utf-8') as md_file:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):  # Only include Python files
                    file_path = os.path.join(root, file)
                    code = extract_code_from_file(file_path)
                    
                    # Write the title (file name) and the code to the markdown file
                    md_file.write(f"## {file}\n\n")
                    md_file.write("```python\n")
                    md_file.write(code)
                    md_file.write("\n```\n\n")

# Define the root directory and the output markdown file
root_directory = '.'  # Current directory (change this if needed)
output_markdown = 'all_code.md'

# Create the markdown file with the Python code
create_markdown_from_code(root_directory, output_markdown)

print(f"Markdown file '{output_markdown}' has been created with all code.")
