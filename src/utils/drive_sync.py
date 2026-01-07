"""
Google Drive sync utility using PyDrive.

Usage:
    python -m src.utils.drive_sync --upload --source data/
    python -m src.utils.drive_sync --download --dest data/ --remote datasets
"""
import os
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Try importing PyDrive
try:
    from pydrive.drive import GoogleDrive
    from pydrive.auth import GoogleAuth
    PYDRIVE_AVAILABLE = True
except ImportError:
    PYDRIVE_AVAILABLE = False

console = Console()

# Remote folder structure in Google Drive
DRIVE_ROOT = "aviation-research"
REMOTE_FOLDERS = {
    "datasets": "datasets",
    "models": "models",
    "outputs": "outputs",
    "backups": "backups",
}


class DriveSync:
    """Sync files with Google Drive."""

    def __init__(self, secrets_path: Optional[Path] = None):
        """Initialize Drive sync."""
        if not PYDRIVE_AVAILABLE:
            console.print("[red]PyDrive not installed. Run: pip install PyDrive[/red]")
            console.print("[yellow]Or use rclone instead (see GOOGLE_DRIVE_SETUP.md)[/yellow]")
            sys.exit(1)

        self.secrets_path = secrets_path or Path(__file__).parent.parent.parent / "config" / "client_secrets.json"
        self.drive = None
        self._auth()

    def _auth(self):
        """Authenticate with Google Drive."""
        if not self.secrets_path.exists():
            console.print(f"[red]Client secrets not found: {self.secrets_path}[/red]")
            console.print("\n[yellow]Setup instructions:[/yellow]")
            console.print("1. Go to https://console.cloud.google.com/")
            console.print("2. Create project + enable Drive API")
            console.print("3. Create OAuth credentials (Desktop app)")
            console.print("4. Download client_secrets.json")
            console.print(f"5. Save to: {self.secrets_path}")
            sys.exit(1)

        # Save current directory
        settings_file = Path(__file__).parent.parent.parent / "config" / "my_drive_settings.yaml"

        # Create minimal settings
        if not settings_file.exists():
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_file, "w") as f:
                f.write(f"""
client_config_backend: settings
client_config:
  client_id: null
  client_secret: null

save_credentials: True
save_credentials_backend: file
save_credentials_file: {settings_file.parent / "drive_credentials.json"}

oauth_scope:
  - https://www.googleapis.com/auth/drive

get_refresh_token: True
""")

        # Authenticate
        gauth = GoogleAuth(settings_file=str(settings_file))
        gauth.LocalWebserverAuth()  # Opens browser
        self.drive = GoogleDrive(gauth)
        console.print("[green]✓ Authenticated with Google Drive[/green]")

    def _get_or_create_folder(self, folder_name: str, parent_id: str = "root") -> str:
        """Get folder ID or create if not exists."""
        # Search for folder
        query = f"'{parent_id}' in parents and title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        file_list = self.drive.ListFile({"q": query}).GetList()

        if file_list:
            return file_list[0]["id"]

        # Create folder
        folder = self.drive.CreateFile(
            {"title": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [{"id": parent_id}]}
        )
        folder.Upload()
        return folder["id"]

    def _get_drive_root_id(self) -> str:
        """Get or create root folder."""
        return self._get_or_create_folder(DRIVE_ROOT)

    def upload_file(self, local_path: Path, remote_folder: str = "datasets", progress_callback=None) -> bool:
        """Upload a file to Google Drive."""
        if not local_path.exists():
            console.print(f"[red]File not found: {local_path}[/red]")
            return False

        # Get/create remote folder
        root_id = self._get_drive_root_id()
        folder_id = self._get_or_create_folder(remote_folder, root_id)

        # Create file
        file = self.drive.CreateFile(
            {"title": local_path.name, "parents": [{"id": folder_id}]}
        )

        # Upload with progress
        console.print(f"[cyan]Uploading {local_path.name}...[/cyan]")
        file.SetContentFile(str(local_path))

        try:
            file.Upload()
            console.print(f"[green]✓ Uploaded: {local_path.name}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]✗ Upload failed: {e}[/red]")
            return False

    def upload_folder(self, local_folder: Path, remote_folder: str = "datasets") -> int:
        """Upload all files from a folder."""
        if not local_folder.exists():
            console.print(f"[red]Folder not found: {local_folder}[/red]")
            return 0

        # Get all files
        files = [f for f in local_folder.rglob("*") if f.is_file()]
        # Filter out common junk
        files = [f for f in files if f.name not in [".DS_Store", ".gitkeep", "Thumbs.db"]]

        if not files:
            console.print("[yellow]No files to upload[/yellow]")
            return 0

        console.print(f"[cyan]Found {len(files)} files to upload[/cyan]")

        success = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Uploading...", total=len(files))

            for file in files:
                # Calculate relative path for subfolder structure
                rel_path = file.relative_to(local_folder)
                remote_subfolder = remote_folder

                # Create nested folder structure if needed
                if len(rel_path.parts) > 1:
                    remote_subfolder = f"{remote_folder}/{'/'.join(rel_path.parts[:-1])}"

                if self.upload_file(file, remote_subfolder):
                    success += 1
                progress.update(task, advance=1)

        console.print(f"[green]✓ Uploaded {success}/{len(files)} files[/green]")
        return success

    def download_file(self, file_id: str, local_path: Path) -> bool:
        """Download a file from Google Drive."""
        try:
            file = self.drive.CreateFile({"id": file_id})
            file.GetContentFile(str(local_path))
            return True
        except Exception as e:
            console.print(f"[red]✗ Download failed: {e}[/red]")
            return False


app = typer.Typer()


@app.command()
def upload(
    source: str = typer.Argument("data/", help="Local folder to upload"),
    remote: str = typer.Option("datasets", help="Remote folder name"),
):
    """Upload files to Google Drive."""
    sync = DriveSync()
    local_path = Path(source)

    if local_path.is_file():
        sync.upload_file(local_path, remote)
    else:
        sync.upload_folder(local_path, remote)


@app.command()
def download(
    dest: str = typer.Argument("data/", help="Local destination folder"),
    remote: str = typer.Option("datasets", help="Remote folder name"),
):
    """Download files from Google Drive."""
    console.print("[yellow]Download feature coming soon[/yellow]")
    console.print("[yellow]For now, use rclone:[/yellow]")
    console.print(f"  rclone copy gdrive:aviation-research/{remote}/ {dest} -P")


@app.command()
def test():
    """Test Google Drive connection."""
    sync = DriveSync()
    console.print("[green]✓ Connection successful![/green]")
    console.print(f"Root folder: {DRIVE_ROOT}")


if __name__ == "__main__":
    if not PYDRIVE_AVAILABLE:
        console.print("[red]PyDrive not installed. Run: pip install PyDrive[/red]")
        console.print("[yellow]Or use rclone (see GOOGLE_DRIVE_SETUP.md)[/yellow]")
    else:
        app()
