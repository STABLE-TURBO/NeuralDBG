# Neural Aquarium Build System

This directory contains build configuration files for packaging Neural Aquarium as a desktop application using Electron Builder.

## Contents

- **electron-builder.yml**: Main Electron Builder configuration
- **entitlements.mac.plist**: macOS entitlements for app sandbox and notarization
- **notarize.js**: Script for macOS notarization
- **dev-app-update.yml**: Development auto-update configuration
- **CODE_SIGNING.md**: Comprehensive code signing setup guide
- **linux/**: Linux-specific scripts (post-install, post-remove)

## Building the Application

### Prerequisites

```bash
cd neural/aquarium
npm install
```

### Development Build

```bash
# Run in development mode with hot reload
npm run electron:dev
```

### Production Build

```bash
# Build for current platform
npm run electron:build

# Build for specific platforms
npm run electron:build:win    # Windows (NSIS + MSI)
npm run electron:build:mac    # macOS (DMG + ZIP)
npm run electron:build:linux  # Linux (AppImage + deb + rpm)

# Build for all platforms
npm run electron:build:all
```

### Release Build with Auto-Update

```bash
# Requires GH_TOKEN environment variable
export GH_TOKEN=your_github_token
npm run release
```

## Platform-Specific Builds

### Windows

**Outputs:**
- `Neural Aquarium-{version}-win-x64.exe` - NSIS installer (recommended)
- `Neural Aquarium-{version}-win-x64.msi` - MSI installer
- `Neural Aquarium-{version}-win-arm64.exe` - NSIS installer (ARM64)
- `Neural Aquarium-{version}-win-arm64.msi` - MSI installer (ARM64)

**Requirements:**
- Windows 10+
- Code signing certificate (optional but recommended)

**NSIS Features:**
- Custom installation directory
- Desktop and Start Menu shortcuts
- Automatic updates
- Uninstaller
- Per-user installation (no admin required)

### macOS

**Outputs:**
- `Neural Aquarium-{version}-mac-x64.dmg` - Intel Macs
- `Neural Aquarium-{version}-mac-arm64.dmg` - Apple Silicon
- `Neural Aquarium-{version}-mac-universal.dmg` - Universal binary
- `Neural Aquarium-{version}-mac-{arch}.zip` - ZIP archives

**Requirements:**
- macOS 10.13+
- Apple Developer account (for signing and notarization)
- Xcode Command Line Tools

**Features:**
- DMG with drag-to-Applications
- Code signed and notarized
- Hardened runtime
- Automatic updates

### Linux

**Outputs:**
- `Neural Aquarium-{version}-linux-x64.AppImage` - Universal Linux binary
- `Neural Aquarium-{version}-linux-x64.deb` - Debian/Ubuntu
- `Neural Aquarium-{version}-linux-x64.rpm` - Fedora/RHEL
- `Neural Aquarium-{version}-linux-arm64.{AppImage,deb,rpm}` - ARM64 variants

**Requirements:**
- Ubuntu 18.04+ / Fedora 32+ or equivalent
- GPG key (for package signing)

**Features:**
- Desktop integration (.desktop file)
- MIME type association (.neural files)
- Icon installation
- Post-install/remove scripts

## Auto-Update Configuration

Neural Aquarium uses `electron-updater` for automatic updates.

### Configuration

The auto-update provider is configured in `package.json`:

```json
"build": {
  "publish": [
    {
      "provider": "github",
      "owner": "neural-dsl",
      "repo": "neural-dsl",
      "releaseType": "release"
    }
  ]
}
```

### Update Flow

1. Application checks for updates on startup and every hour
2. If update is available, downloads in background
3. User is notified when download completes
4. User can install immediately or postpone
5. Application restarts to apply update

### Testing Updates

For development testing:

```bash
# Set update server URL
export ELECTRON_UPDATE_URL=http://localhost:8080/releases

# Run dev app update server
python -m http.server 8080 --directory dist/
```

## Code Signing

See [CODE_SIGNING.md](./CODE_SIGNING.md) for detailed instructions on:

- Windows code signing certificates
- macOS Developer ID and notarization
- Linux GPG package signing
- CI/CD integration
- Security best practices

## CI/CD Integration

GitHub Actions workflow at `.github/workflows/aquarium-release.yml` automates:

1. Building for all platforms (Windows, macOS, Linux)
2. Code signing on each platform
3. Generating checksums
4. Creating GitHub releases
5. Publishing installers as release assets
6. Configuring auto-update

### Triggering a Release

**Option 1: Tag Push**
```bash
git tag aquarium-v0.3.0
git push origin aquarium-v0.3.0
```

**Option 2: Manual Workflow Dispatch**
- Go to Actions tab in GitHub
- Select "Aquarium Release" workflow
- Click "Run workflow"
- Enter version number (e.g., 0.3.0)

## Troubleshooting

### Build Fails on macOS

**Issue:** "Code signing failed"
- Ensure certificate is in Keychain
- Check `security find-identity -v -p codesigning`
- Verify entitlements file exists

**Issue:** "Notarization failed"
- Check Apple ID and app-specific password
- Verify Team ID is correct
- Wait 10-15 minutes for Apple servers

### Build Fails on Windows

**Issue:** "Certificate not found"
- Check WIN_CSC_LINK path
- Verify certificate password
- Ensure certificate is valid

**Issue:** "Timestamp server unavailable"
- Network issue, retry
- Try different timestamp server

### Build Fails on Linux

**Issue:** "Missing dependencies"
```bash
# Ubuntu/Debian
sudo apt-get install -y rpm fakeroot dpkg

# Fedora
sudo dnf install -y rpm-build dpkg
```

### Auto-Update Not Working

1. Check GitHub release is published (not draft)
2. Verify `publish` config in package.json
3. Check GH_TOKEN has repo scope
4. Look at Electron logs: `~/.config/Neural Aquarium/logs/`

## Directory Structure

```
build/
├── electron-builder.yml        # Main builder config
├── entitlements.mac.plist     # macOS entitlements
├── notarize.js                # macOS notarization script
├── dev-app-update.yml         # Dev update config
├── CODE_SIGNING.md            # Signing guide
├── README.md                  # This file
├── linux/
│   ├── after-install.sh       # Post-install script
│   └── after-remove.sh        # Post-remove script
├── icon.ico                   # Windows icon
├── icon.icns                  # macOS icon
├── icon.png                   # Linux icon
├── file-icon.ico              # .neural file icon
├── background.png             # DMG background
└── icons/                     # Linux icon set
    ├── 16x16.png
    ├── 32x32.png
    ├── 48x48.png
    ├── 64x64.png
    ├── 128x128.png
    ├── 256x256.png
    └── 512x512.png
```

## Resources

- [Electron Builder Documentation](https://www.electron.build/)
- [electron-updater Documentation](https://www.electron.build/auto-update)
- [Apple Notarization Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Windows Code Signing](https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)
