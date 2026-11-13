#!/bin/bash
# Build Haru.app and the command-line client with swiftc; Xcode is not needed.
#   PYTHON=/path/to/python ./build_app.sh   # default: ../.venv/bin/python
# PYTHON is the interpreter that has haru_mlx installed; the app starts the
# model server with it.
set -euo pipefail
cd "$(dirname "$0")"
if ! xcrun --find swiftc >/dev/null 2>&1; then
    echo "swiftc not found: install the Command Line Tools with 'xcode-select --install' (or Xcode)." >&2
    exit 1
fi
PYTHON="${PYTHON:-$(cd .. && pwd)/.venv/bin/python}"
if ! "$PYTHON" -c "import haru_mlx" >/dev/null 2>&1; then
    echo "haru_mlx is not installed for $PYTHON: see README.md (or set PYTHON=...)." >&2
    exit 1
fi
APP=build/Haru.app
# Apple silicon only (MLX); macOS 13 or later, matching LSMinimumSystemVersion.
TARGET=(-target arm64-apple-macos13.0)
FRAMEWORKS=(-framework AudioToolbox -framework CoreAudio)

mkdir -p .build
swiftc -O "${TARGET[@]}" Shared/*.swift CLI/main.swift -o .build/haru-client "${FRAMEWORKS[@]}"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
swiftc -O "${TARGET[@]}" -parse-as-library Shared/*.swift App/*.swift -o "$APP/Contents/MacOS/Haru" \
    "${FRAMEWORKS[@]}" -framework SwiftUI -framework AppKit -framework AVFoundation -framework Accelerate -framework CoreMedia -framework CoreVideo

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Haru</string>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleLocalizations</key>
    <array><string>en</string><string>ja</string><string>zh-Hans</string><string>ko</string><string>es</string><string>fr</string><string>de</string><string>pt</string><string>ru</string></array>
    <key>CFBundleDisplayName</key><string>Haru</string>
    <key>CFBundleIdentifier</key><string>ai.higurashi.haru</string>
    <key>CFBundleExecutable</key><string>Haru</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Haru uses the microphone to pass your voice to the model.</string>
    <key>HaruPython</key><string>$PYTHON</string>
</dict>
</plist>
EOF
# The icon, and the microphone prompt in each language (the UI's own strings are in App/Strings.swift).
mkdir -p "$APP/Contents/Resources"
cp Resources/AppIcon.icns "$APP/Contents/Resources/"
cp -R Resources/*.lproj "$APP/Contents/Resources/"
codesign --force --sign - "$APP"
echo "built $APP and .build/haru-client"
