import os
import subprocess
from pathlib import Path

def generate_audit_dump():
    # Configuration
    output_filename = "project_audit_dump.txt"
    project_root = Path(__file__).parent.parent.absolute()
    output_path = project_root / output_filename
    
    # Exclude patterns (even if not in .gitignore)
    exclude_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.pyc', '.so', '.exe', '.bin'}
    exclude_files = {output_filename, "package-lock.json", "Cargo.lock"}

    print(f"Generating audit dump at: {output_path}")

    try:
        # Use git ls-files to get all tracked files (respects .gitignore)
        result = subprocess.run(
            ["git", "ls-files"], 
            cwd=project_root, 
            capture_output=True, 
            text=True, 
            check=True
        )
        files = result.stdout.splitlines()
    except Exception as e:
        print(f"Git not found or error: {e}. Falling back to manual walk.")
        files = []
        for root, _, filenames in os.walk(project_root):
            for filename in filenames:
                rel_path = os.path.relpath(os.path.join(root, filename), project_root)
                if not rel_path.startswith(".") and "node_modules" not in rel_path:
                    files.append(rel_path)

    count = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"PROJECT AUDIT DUMP - GENERATED ON {os.uname().nodename}\n")
        outfile.write(f"BASE DIRECTORY: {project_root}\n")
        outfile.write("=" * 80 + "\n\n")

        for rel_path in sorted(files):
            file_path = project_root / rel_path
            
            # Skip binary files and excluded files
            if file_path.suffix.lower() in exclude_extensions or rel_path in exclude_files:
                continue
            
            if not file_path.is_file():
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as infile:
                    content = infile.read()
                    
                outfile.write("=" * 80 + "\n")
                outfile.write(f"FILE: {rel_path}\n")
                outfile.write("=" * 80 + "\n")
                outfile.write(content)
                outfile.write("\n\n")
                count += 1
                print(f"Added: {rel_path}")
            except Exception as e:
                print(f"Could not read {rel_path}: {e}")

    print(f"\nDone! {count} files aggregated into {output_filename}")

if __name__ == "__main__":
    generate_audit_dump()
