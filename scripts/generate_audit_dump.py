import os
import subprocess
from pathlib import Path

def is_binary(file_path):
    """Check if file contains null bytes or is likely binary."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception:
        return True

def get_tracked_files(project_root):
    """Get all files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )
        return set(result.stdout.splitlines())
    except Exception as e:
        print(f"Git not found or error: {e}")
        return set()

def get_force_included_files(project_root):
    """Find files in explicitly included directories even if git-ignored."""
    force_include_dirs = ['docs']
    force_included = set()
    for d in force_include_dirs:
        dir_path = project_root / d
        if not dir_path.exists():
            continue
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(project_root).as_posix()
                force_included.add(rel_path)
    return force_included

def generate_audit_dump():
    # Configuration
    output_filename = "project_audit_dump.txt"
    project_root = Path(__file__).parent.parent.absolute()
    output_path = project_root / output_filename

    # Files to always exclude from the dump itself
    exclude_files = {output_filename}
    # Directories to strictly avoid
    exclude_dirs = {'.git', '.venv', 'venv', 'env', '__pycache__', 'node_modules', 'target', 'build', 'dist'}
    
    print(f"Generating audit dump at: {output_path}")

    # 1. Get tracked files (base set)
    files_to_audit = get_tracked_files(project_root)
    
    # 2. Add force-included files (like docs/) that might be ignored by git
    files_to_audit.update(get_force_included_files(project_root))

    # 3. Add top-level configs manually to ensure they are captured
    top_configs = ['pyproject.toml', '.gitignore', 'README.md', 'uv.lock', 'GEMINI.md']
    for config in top_configs:
        if (project_root / config).exists():
            files_to_audit.add(config)

    # 4. Fallback/Safety: If git failed, or to find untracked files in src
    # Only if we have very few files, otherwise we trust git + force_include
    if len(files_to_audit) < 5:
        print("Minimal files found via git, performing manual walk...")
        for root, dirs, filenames in os.walk(project_root):
            # Prune excluded dirs
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for filename in filenames:
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(project_root).as_posix()
                files_to_audit.add(rel_path)

    count = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"PROJECT AUDIT DUMP - GENERATED ON {os.uname().nodename}\n")
        outfile.write(f"BASE DIRECTORY: {project_root}\n")
        outfile.write("=" * 80 + "\n\n")

        for rel_path in sorted(list(files_to_audit)):
            file_path = project_root / rel_path
            
            # Skip if it's the output file itself
            if rel_path in exclude_files:
                continue
                
            # Skip if any part of the path is in exclude_dirs
            if any(part in exclude_dirs for part in Path(rel_path).parts):
                continue
                
            if not file_path.is_file():
                continue

            # Skip large files > 1MB (prevents dumping massive datasets/logs)
            if file_path.stat().st_size > 1024 * 1024:
                print(f"Skipping {rel_path} (too large: {file_path.stat().st_size / 1024:.1f}KB)")
                continue

            # Skip binary files
            if is_binary(file_path):
                print(f"Skipping {rel_path} (binary)")
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
