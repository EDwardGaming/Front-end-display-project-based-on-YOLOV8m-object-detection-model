import os
import subprocess
import sys

# Size limit in bytes (50 MiB)
SIZE_LIMIT = 50 * 1024 * 1024
GIT_DIR = ".git"
EXCLUDE_FILE = os.path.join(GIT_DIR, "info", "exclude")

def run_command(command):
    """Runs a shell command, prints its output, and returns its result object."""
    try:
        # Using UTF-8 encoding is important for handling file paths and commit messages
        result = subprocess.run(
            command, 
            check=True, 
            text=True, 
            capture_output=True, 
            encoding='utf-8'
        )
        # Print stdout if it's not empty
        if result.stdout:
            print(result.stdout.strip())
        # Print stderr if it's not empty (e.g., for git push progress)
        if result.stderr:
            print(result.stderr.strip())
        return result
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is Git installed and in your PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # Print a more informative error message
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return Code: {e.returncode}")
        print(f"Stdout: {e.stdout.strip()}")
        print(f"Stderr: {e.stderr.strip()}")
        return None  # Indicate failure

def is_file_tracked(file_path):
    """Check if a file is tracked by Git using ls-files."""
    # `git ls-files --error-unmatch` returns 0 if tracked, 1 if not.
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", file_path],
        capture_output=True, 
        text=True,
        encoding='utf-8'
    )
    return result.returncode == 0

def main():
    """Main function to find large files, manage them, and push to git."""
    # 1. Check if it's a git repository
    if not os.path.isdir(GIT_DIR):
        print("Error: Please run this script from the root of a Git repository.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Scanning for files larger than {SIZE_LIMIT / 1024 / 1024} MiB ---")

    large_files_found = []
    
    # 2. Walk the directory tree to find large files
    for root, dirs, files in os.walk("."):
        # Don't scan the .git directory itself
        if GIT_DIR in dirs:
            dirs.remove(GIT_DIR)
            
        for name in files:
            file_path = os.path.join(root, name)
            # Use a try-except block to handle potential permission errors
            try:
                if os.path.getsize(file_path) > SIZE_LIMIT:
                    # Use normalized relative paths
                    relative_path = os.path.normpath(file_path)
                    large_files_found.append(relative_path)
            except OSError as e:
                print(f"Warning: Could not access {file_path}: {e}", file=sys.stderr)

    # 3. Process the large files found
    if not large_files_found:
        print("No large files found.")
    else:
        # Ensure the .git/info directory exists before trying to write to it
        os.makedirs(os.path.dirname(EXCLUDE_FILE), exist_ok=True)
        
        # Read existing rules from the exclude file to avoid duplicates
        try:
            with open(EXCLUDE_FILE, "r", encoding="utf-8") as f:
                excluded_files = set(line.strip() for line in f)
        except FileNotFoundError:
            excluded_files = set()

        for file_path in large_files_found:
            # Use forward slashes for Git's path format
            git_path = file_path.replace(os.sep, '/')
            print(f"Found large file: {git_path}")

            # If the file is already tracked by Git, untrack it
            if is_file_tracked(git_path):
                print(f"  - File is tracked. Removing from index with 'git rm --cached'.")
                run_command(["git", "rm", "--cached", git_path])

            # Add the file to the exclude list if it's not already there
            if git_path not in excluded_files:
                print(f"  - Adding to {EXCLUDE_FILE}")
                with open(EXCLUDE_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n{git_path}")
    
    # 4. Execute the Git push sequence
    print("\n--- Starting Git push sequence ---")
    
    print("Running 'git add .'")
    run_command(["git", "add", "."])
    
    # Check git status to see if there are any changes to commit
    status_result = run_command(["git", "status", "--porcelain"])
    if status_result and status_result.stdout:
        print("Changes detected. Running 'git commit'.")
        run_command(["git", "commit", "-m", "Auto push: Filter large files and update index"])
        
        print("Running 'git push origin master'...")
        push_result = run_command(["git", "push", "origin", "master"])
        if push_result:
            print("Push successful.")
    else:
        print("No changes to commit.")

    print("--- Push filter script finished ---")


if __name__ == "__main__":
    # On Windows, Python might default to a different encoding for console output.
    # This ensures that UTF-8 characters in file paths are printed correctly.
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    main()
