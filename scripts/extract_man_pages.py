import os
import subprocess

# Define the list of commands
commands = [
    "ls", "cp", "mv", "rm", "mkdir", "rmdir", "find", "tree",
    "cat", "less", "more", "head", "tail", "nano", "vim",
    "tar", "gzip", "gunzip", "zip", "unzip", "xz",
    "chmod", "chown", "chgrp", "umask",
    "ssh", "scp", "rsync", "sftp", "curl", "wget",
    "ps", "top", "htop", "kill", "pkill", "nice", "renice",
    "df", "du", "free", "uptime", "vmstat",
    "apt", "dpkg",
    "adduser", "deluser", "passwd", "usermod", "whoami",
    "who", "w", "id", "last",
    "bash", "sh", "echo", "env", "grep", "sed", "awk", "cut", "sort", "uniq", "wc",
    "cron", "crontab", "at",
    "journalctl", "dmesg", "logrotate",
    "uname", "hostname", "lsb_release",
    "rsync", "dd"
]

# Directory to save MAN pages
output_dir = "/home/ai_admin/my_ai_project/knowledge_base/man_pages"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to extract and save MAN pages
def extract_man_page(command):
    try:
        # Get the MAN page using subprocess and 'col -b' to remove backspaces
        man_output = subprocess.check_output(f"man {command} | col -b", shell=True, text=True)
        # Save to a file in the output directory
        with open(os.path.join(output_dir, f"{command}.txt"), "w") as file:
            file.write(man_output)
        print(f"MAN page for '{command}' saved successfully.")
    except subprocess.CalledProcessError:
        print(f"No MAN page found for '{command}'.")

# Extract MAN pages for each command
for cmd in commands:
    extract_man_page(cmd)

print("MAN page extraction completed.")
