import re
import requests
import paramiko
from dotenv import load_dotenv
import os
load_dotenv()

# Configuration
FILE_URL = "https://www.cc.iaa.es/system/nodes/status"  # Replace with the actual URL
LOCAL_FILE = "status.html"
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")
COMMAND_TO_RUN = "conda activate ctlearn_pytorch && cd neutron_remote_connection && ls"  # Replace with the actual command

COMMANDS_FILE = "commands.txt"
STATE_FILE = "last_command_index.txt"

def get_next_command():
    """Reads the next command to execute from the file, keeping track of progress."""
    if not os.path.exists(COMMANDS_FILE):
        print("Commands file not found.")
        return None
    
    with open(COMMANDS_FILE, "r") as f:
        commands = f.readlines()
    
    if not commands:
        print("No commands to execute.")
        return None
    
    last_index = 0
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            last_index = int(f.read().strip())
    
    if last_index >= len(commands):
        print("All commands have been executed.")
        return None
    
    next_command = commands[last_index].strip()
    
    with open(STATE_FILE, "w") as f:
        f.write(str(last_index + 1))
    
    return next_command

def download_file():
    """ Downloads the HTML file using requests without SSL verification """
    try:
        response = requests.get(FILE_URL, verify=False)  # Disable SSL verification
        with open(LOCAL_FILE, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully (SSL verification disabled).")
    except Exception as e:
        print(f"Error downloading the file: {e}")

def parse_gpu_usage():
    """ Parses the HTML file to find GPUs with usage below 20% """
    low_usage_nodes = []
    with open(LOCAL_FILE, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Regular expression to extract node names and GPU usage
    pattern = re.compile(
        r'<td[^>]*><a href="ssh://(.*?)">.*?</a></td>\s*'  # Node (e.g., n1.iaa.es)
        r'<td[^>]*>.*?</td>\s*'  # %CPU
        r'<td[^>]*>.*?</td>\s*'  # Cores
        r'<td[^>]*>.*?</td>\s*'  # %Mem
        r'<td[^>]*>.*?</td>\s*'  # GBMem
        r'<td[^>]*>([\d.]+)</td>',  # %GPU
        re.DOTALL
    )

    for match in pattern.findall(content):
        node, gpu_usage = match[0], float(match[1])
        if gpu_usage < 20:
            low_usage_nodes.append(node)
    
    return low_usage_nodes

def execute_remote_command(node):
    """ Connects to the node via SSH and executes a command """
    command = get_next_command()
    try:
        print(f"Connecting to {node} to execute {command}...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(node, username=SSH_USER, password=SSH_PASSWORD)
        stdin, stdout, stderr = client.exec_command(command)
        print(f"Command output on {node}:\n", stdout.read().decode())
        client.close()
    except Exception as e:
        print(f"SSH connection error on {node}: {e}")

def main():
    download_file()
    idle_gpus = parse_gpu_usage()
    
    if idle_gpus:
        print(f"Detected GPUs with low usage on nodes: {idle_gpus}")
        for node in idle_gpus:
            execute_remote_command(node)
            print(node)
    else:
        print("No GPUs with usage below 20% were found.")

if __name__ == "__main__":
    main()