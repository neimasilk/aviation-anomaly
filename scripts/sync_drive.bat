@echo off
REM sync_drive.bat - Sync dengan Google Drive menggunakan rclone (Windows)
REM
REM Penggunaan:
REM   .\scripts\sync_drive.bat upload
REM   .\scripts\sync_drive.bat download
REM   .\scripts\sync_drive.bat download-data    - Hanya download data (tanpa model)
REM   .\scripts\sync_drive.bat secrets

setlocal

set DRIVE_REMOTE=gdrive
set DRIVE_ROOT=aviation-research
set ACTION=%1

if "%ACTION%"=="" set ACTION=download

echo ========================================
echo Google Drive Sync: %ACTION%
echo ========================================

if "%ACTION%"=="upload" goto upload
if "%ACTION%"=="download" goto download
if "%ACTION%"=="download-data" goto download_data
if "%ACTION%"=="secrets" goto secrets
if "%ACTION%"=="upload-secrets" goto upload_secrets

goto usage

:upload
echo.
echo [UPLOAD] Uploading data...
rclone copy data\ %DRIVE_REMOTE%:%DRIVE_ROOT%/datasets/ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo [UPLOAD] Uploading models...
rclone copy models\ %DRIVE_REMOTE%:%DRIVE_ROOT%/models/ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo [UPLOAD] Uploading outputs...
rclone copy outputs\ %DRIVE_REMOTE%:%DRIVE_ROOT%/outputs/ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo ✅ Upload complete!
goto end

:download_data
echo.
echo [DOWNLOAD] Downloading data ONLY (no models)...
rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/datasets/ data\ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo [DOWNLOAD] Downloading outputs...
rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/outputs/ outputs\ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo ✅ Download complete (models skipped to save SSD space)!
echo.
echo 💡 To download specific models manually:
echo    rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/models/001 models/001
echo    rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/models/002 models/002
goto end

:download
echo.
echo [DOWNLOAD] Downloading data...
rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/datasets/ data\ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo [DOWNLOAD] Downloading models...
rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/models/ models\ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo [DOWNLOAD] Downloading outputs...
rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/outputs/ outputs\ ^
    --exclude="**/.gitkeep" ^
    --exclude="**/.DS_Store" ^
    --progress

echo.
echo ✅ Download complete!
goto end

:secrets
echo.
echo [SECRETS] Downloading secrets from Drive...
if not exist config mkdir config

rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/secrets/.env . --progress 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ .env downloaded
) else (
    echo ⚠️  .env not found on Drive (first time setup?)
)

rclone copy %DRIVE_REMOTE%:%DRIVE_ROOT%/secrets/ config\ ^
    --include="client_secrets.json" ^
    --include="api_keys.json" ^
    --progress

echo.
echo ✅ Secrets downloaded!
goto end

:upload_secrets
echo.
echo [SECRETS] Uploading secrets to Drive...

if exist .env (
    rclone copy .env %DRIVE_REMOTE%:%DRIVE_ROOT%/secrets/ --progress
    echo ✅ .env uploaded
)

if exist config\ (
    rclone copy config\ %DRIVE_REMOTE%:%DRIVE_ROOT%/secrets/ ^
        --include="client_secrets.json" ^
        --include="api_keys.json" ^
        --include="my_drive_settings.yaml" ^
        --progress
)

echo.
echo ✅ Secrets uploaded!
echo ⚠️  Remember: .env is still in .gitignore, won't be committed
goto end

:usage
echo.
echo Usage: sync_drive.bat {upload^|download^|download-data^|secrets^|upload-secrets}
echo.
echo Commands:
echo   upload         - Upload data/models/outputs to Drive
echo   download       - Download data/models/outputs from Drive
echo   download-data  - Download ONLY data (skip models to save SSD)
echo   secrets        - Download .env and secrets from Drive
echo   upload-secrets - Upload .env and secrets to Drive
exit /b 1

:end
endlocal
