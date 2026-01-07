#!/bin/bash
# sync_drive.sh - Sync dengan Google Drive menggunakan rclone
#
# Penggunaan:
#   ./scripts/sync_drive.sh upload    # Upload ke Drive
#   ./scripts/sync_drive.sh download  # Download dari Drive
#   ./scripts/sync_drive.sh secrets   # Download secrets (API keys)

set -e

DRIVE_REMOTE="gdrive"
DRIVE_ROOT="aviation-research"

ACTION=${1:-"download"}

echo "🚀 Google Drive Sync: $ACTION"

case $ACTION in
    upload)
        echo "📦 Uploading data..."
        rclone copy data/ "$DRIVE_REMOTE:$DRIVE_ROOT/datasets/" \
            --exclude="**/.gitkeep" \
            --exclude="**/.DS_Store" \
            --progress

        echo "🤖 Uploading models..."
        rclone copy models/ "$DRIVE_REMOTE:$DRIVE_ROOT/models/" \
            --exclude="**/.gitkeep" \
            --exclude="**/.DS_Store" \
            --progress

        echo "📊 Uploading outputs..."
        rclone copy outputs/ "$DRIVE_REMOTE:$DRIVE_ROOT/outputs/" \
            --exclude="**/.gitkeep" \
            --exclude="**/.DS_Store" \
            --progress

        echo "✅ Upload complete!"
        ;;

    download)
        echo "📥 Downloading data..."
        rclone copy "$DRIVE_REMOTE:$DRIVE_ROOT/datasets/" data/ \
            --exclude="**/.gitkeep" \
            --exclude="**/.DS_Store" \
            --progress

        echo "🤖 Downloading models..."
        rclone copy "$DRIVE_REMOTE:$DRIVE_ROOT/models/" models/ \
            --exclude="**/.gitkeep" \
            --exclude="**/.DS_Store" \
            --progress

        echo "📊 Downloading outputs..."
        rclone copy "$DRIVE_REMOTE:$DRIVE_ROOT/outputs/" outputs/ \
            --exclude="**/.gitkeep" \
            --exclude="**/.DS_Store" \
            --progress

        echo "✅ Download complete!"
        ;;

    secrets)
        echo "🔐 Downloading secrets from Drive..."
        mkdir -p config

        # Download .env file
        if rclone copy "$DRIVE_REMOTE:$DRIVE_ROOT/secrets/.env" . \
            --progress 2>/dev/null; then
            echo "✅ .env downloaded"
        else
            echo "⚠️  .env not found on Drive (first time setup?)"
        fi

        # Download other secrets if exist
        rclone copy "$DRIVE_REMOTE:$DRIVE_ROOT/secrets/" config/ \
            --include="client_secrets.json" \
            --include="api_keys.json" \
            --progress

        echo "✅ Secrets downloaded!"
        ;;

    upload-secrets)
        echo "🔐 Uploading secrets to Drive..."
        mkdir -p "$DRIVE_REMOTE:$DRIVE_ROOT/secrets"

        # Upload .env
        if [ -f .env ]; then
            rclone copy .env "$DRIVE_REMOTE:$DRIVE_ROOT/secrets/" --progress
            echo "✅ .env uploaded"
        fi

        # Upload other secrets
        if [ -d config ]; then
            rclone copy config/ "$DRIVE_REMOTE:$DRIVE_ROOT/secrets/" \
                --include="client_secrets.json" \
                --include="api_keys.json" \
                --include="my_drive_settings.yaml" \
                --progress
        fi

        echo "✅ Secrets uploaded!"
        echo "⚠️  Remember: .env is still in .gitignore, won't be committed"
        ;;

    *)
        echo "Usage: $0 {upload|download|secrets|upload-secrets}"
        echo ""
        echo "Commands:"
        echo "  upload         - Upload data/models/outputs to Drive"
        echo "  download       - Download data/models/outputs from Drive"
        echo "  secrets        - Download .env and secrets from Drive"
        echo "  upload-secrets - Upload .env and secrets to Drive"
        exit 1
        ;;
esac
