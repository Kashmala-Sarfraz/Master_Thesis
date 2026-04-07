# loads envorinment variables or paths 
import os
# Ensure we use a plaintext keyring backend for simplicity
os.environ["PYTHON_KEYRING_BACKEND"] = "keyrings.alt.file.PlaintextKeyring"
# library for command line argument parsing, used in __main__ block
import argparse
# library for securely getting password input since getpass hides input in terminal
import getpass
# library for using @dataclass which simplifies class creation and skipping boilerplate (repetitive) code
from dataclasses import dataclass
# library for handling filesystem paths in an OS-independent way
# Path also has useful methods for reading/writing small files like exists(), read_text(), write_text(), unlink()
from pathlib import Path
# library for securely storing and retrieving passwords
# here we use get_password(service_name, username),
# set_password(service_name, username, password) and delete_password(service_name, username)
import keyring
# one of the parameters for keyring functions
SERVICE_NAME = "WRDS"
# this is a file path we join the home directory with a filename with /
# with a dot prefix to make it hidden on Unix systems
# so when we look for with "ls" we won't see it unless we use "ls -a" command
LAST_USER_FILE = Path.home() / ".wrds_user"  # remembers last username

# with @dataclass we replace this boilerplate code:
# class Credentials:
#    def __init__(self, username: str, password: str):
#        self.username = username
#        self.password = password
# with this below:
@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


def get_wrds_credentials() -> Credentials:
    """
    Automatically retrieves credentials for wrds.
    - On first run: asks for username and password/token, stores them.
    - On later runs: loads both silently from the system keyring.
    """
    # Try to remember the last-used username
    if LAST_USER_FILE.exists():
        username = LAST_USER_FILE.read_text().strip()
    else:
        username = input(f"Username for {SERVICE_NAME}: ").strip()
        LAST_USER_FILE.write_text(username)

    # Try to retrieve the stored password for this username
    password = keyring.get_password(SERVICE_NAME, username)

    # If not found, prompt once and store it securely
    if not password:
        password = getpass.getpass(f"Password or token for {username} at {SERVICE_NAME}: ")
        keyring.set_password(SERVICE_NAME, username, password)
        print(f"Stored credentials for '{username}' in keyring under '{SERVICE_NAME}'")

    return Credentials(username, password)


def reset_credentials(full_reset: bool = False):
    """
    Clears stored username and optionally removes password from keyring.
    """
    if LAST_USER_FILE.exists():
        username = LAST_USER_FILE.read_text().strip()
        LAST_USER_FILE.unlink()
        print(f"Removed stored username '{username}'")

        if full_reset:
            try:
                keyring.delete_password(SERVICE_NAME, username)
                print(f"Deleted password for '{username}' from keyring under '{SERVICE_NAME}'")
            except keyring.errors.PasswordDeleteError:
                print(f"No keyring entry found for '{username}'")

    else:
        print("No stored username found — nothing to reset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage stored wrds credentials.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove both stored username and password from keyring.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_credentials(full_reset=args.reset)
    else:
        creds = get_wrds_credentials()
        print(f"Using credentials for '{creds.username}'")
