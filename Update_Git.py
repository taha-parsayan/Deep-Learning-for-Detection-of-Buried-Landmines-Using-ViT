import subprocess

def git_command(command):
    """Function to run git commands."""
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        print(f"Error message: {e.stderr}")
        return None

def git_add(file_path):
    """Add file to git."""
    return git_command(["git", "add", file_path])

def git_commit(commit_message):
    """Commit the added files."""
    return git_command(["git", "commit", "-m", commit_message])

def git_push(branch_name):
    """Push the changes to the specified branch."""
    return git_command(["git", "push", "origin", branch_name])

# Example usage
file_to_add = "path_to_your_file"  # Replace with the path to your file
commit_message = "Your commit message"  # Replace with your commit message
branch_name = "your_branch"  # Replace with the branch name (e.g., 'main' or 'langchain')

# Run the git commands
if git_add(file_to_add):
    print(f"Successfully added {file_to_add} to git.")
    
    if git_commit(commit_message):
        print(f"Successfully committed changes with message: '{commit_message}'")
        
        if git_push(branch_name):
            print(f"Successfully pushed changes to branch {branch_name}.")
        else:
            print(f"Failed to push changes to branch {branch_name}.")
    else:
        print("Failed to commit changes.")
else:
    print(f"Failed to add {file_to_add} to git.")
