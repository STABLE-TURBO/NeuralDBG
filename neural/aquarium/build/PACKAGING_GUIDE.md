# Neural Aquarium Packaging & Distribution Guide

Complete guide for packaging and distributing Neural Aquarium as a desktop application.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Prerequisites](#prerequisites)
4. [Building](#building)
5. [Code Signing](#code-signing)
6. [Auto-Updates](#auto-updates)
7. [CI/CD](#cicd)
8. [Testing](#testing)
9. [Release Process](#release-process)
10. [Troubleshooting](#troubleshooting)

## Overview

Neural Aquarium uses Electron Builder to create native installers for:

- **Windows**: NSIS and MSI installers (x64, ARM64)
- **macOS**: DMG and ZIP (Intel, Apple Silicon, Universal)
- **Linux**: AppImage, DEB, and RPM (x64, ARM64)

Auto-updates are powered by `electron-updater` with GitHub Releases as the distribution mechanism.

## Quick Start

```bash
# Navigate to aquarium directory
cd neural/aquarium

# Install dependencies
npm install

# Development mode with hot reload
npm run electron:dev

# Build for current platform
npm run electron:build

# Build for all platforms
npm run electron:build:all
```

## Prerequisites

### All Platforms

- Node.js 18+ and npm
- Python 3.8+ (for backend)
- Git

### Windows (for building Windows apps)

- Windows 10 or later
- Windows SDK
- Optional: Code signing certificate

### macOS (for building macOS apps)

- macOS 10.13 or later
- Xcode Command Line Tools: `xcode-select --install`
- Optional: Apple Developer account ($99/year)

### Linux (for building Linux apps)

- Ubuntu 18.04+ or equivalent
- Additional packages:
  ```bash
  sudo apt-get install -y rpm fakeroot dpkg
  ```

### Cross-Platform Building

Electron Builder supports limited cross-platform building:

- **From macOS**: Can build for macOS, Windows, Linux
- **From Linux**: Can build for Linux, Windows (limited)
- **From Windows**: Can only build for Windows

For full cross-platform support, use CI/CD (GitHub Actions).

## Building

### Directory Structure

```
neural/aquarium/
├── build/               # Build configuration and assets
│   ├── electron-builder.yml
│   ├── entitlements.mac.plist
│   ├── notarize.js
│   ├── icon.ico
│   ├── icon.icns
│   ├── icon.png
│   └── linux/
│       ├── after-install.sh
│       └── after-remove.sh
├── electron/            # Electron main process
│   ├── main.js
│   ├── preload.js
│   └── package.json
├── src/                 # React application
├── package.json         # Main package.json
└── dist/               # Build output (generated)
```

### Build Scripts

```bash
# Development
npm run electron:dev        # Run in dev mode with hot reload

# Production builds
npm run electron:build      # Build for current platform
npm run electron:build:win  # Windows only
npm run electron:build:mac  # macOS only
npm run electron:build:linux # Linux only
npm run electron:build:all  # All platforms (requires platform)

# Release (with auto-update)
npm run release             # Build and publish to GitHub
```

### Build Script (Bash)

```bash
./build/build.sh --platform linux --arch x64 --sign --release
```

Options:
- `-p, --platform`: win, mac, linux, all
- `-a, --arch`: x64, arm64, universal, all
- `-s, --sign`: Enable code signing
- `-r, --release`: Build for release
- `-c, --clean`: Clean before building

### Build Script (PowerShell)

```powershell
.\build\build.ps1 -Platform win -Arch x64 -Sign -Release
```

### Configuration

Main configuration in `package.json` under `build` key. See [electron-builder.yml](./electron-builder.yml) for detailed settings.

Key settings:
- `appId`: Unique application ID
- `productName`: Display name
- `directories`: Build resource locations
- `files`: Files to include in package
- `win/mac/linux`: Platform-specific settings
- `publish`: Auto-update configuration

## Code Signing

Code signing is **essential** for:
- Preventing security warnings
- Enabling auto-updates
- App Store distribution (macOS)
- Enterprise deployment

See [CODE_SIGNING.md](./CODE_SIGNING.md) for complete setup instructions.

### Windows

```bash
export WIN_CSC_LINK=/path/to/certificate.pfx
export WIN_CSC_KEY_PASSWORD=your-password
npm run electron:build:win
```

### macOS

```bash
export APPLE_ID=your@email.com
export APPLE_ID_PASSWORD=xxxx-xxxx-xxxx-xxxx
export APPLE_TEAM_ID=YOUR_TEAM_ID
npm run electron:build:mac
```

### Linux

```bash
export GPG_PASSPHRASE=your-passphrase
npm run electron:build:linux
```

### Skip Signing (Development)

```bash
export CSC_IDENTITY_AUTO_DISCOVERY=false
npm run electron:build
```

## Auto-Updates

Neural Aquarium includes automatic update checking and installation.

### How It Works

1. App checks GitHub Releases for new versions
2. Downloads update in background
3. Notifies user when ready
4. User can install immediately or later
5. App restarts to apply update

### Configuration

Auto-update settings in `package.json`:

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

### Update Channels

- **Release**: Stable releases (default)
- **Beta**: Beta releases (tag with `-beta`)
- **Alpha**: Alpha releases (tag with `-alpha`)

Users can switch channels in settings.

### Testing Updates Locally

```bash
# Build app
npm run electron:build

# Serve dist directory
python -m http.server 8080 --directory dist/

# Set update URL and run app
export ELECTRON_UPDATE_URL=http://localhost:8080/releases
npm run electron
```

### Disabling Auto-Updates

Set in electron/main.js or via user preferences.

## CI/CD

GitHub Actions workflow at `.github/workflows/aquarium-release.yml` automates the entire release process.

### Workflow Triggers

**Automatic (Tag Push):**
```bash
git tag aquarium-v0.3.0
git push origin aquarium-v0.3.0
```

**Manual:**
- Go to Actions > Aquarium Release
- Click "Run workflow"
- Enter version number

### Required Secrets

Add these in GitHub repository settings (Settings > Secrets and variables > Actions):

**Windows:**
- `WIN_CSC_LINK`: Base64-encoded certificate
- `WIN_CSC_KEY_PASSWORD`: Certificate password

**macOS:**
- `APPLE_ID`: Apple ID email
- `APPLE_ID_PASSWORD`: App-specific password
- `APPLE_TEAM_ID`: Team ID
- `APPLE_CERTIFICATE`: Base64-encoded P12
- `APPLE_CERTIFICATE_PASSWORD`: Certificate password

**Linux:**
- `GPG_PRIVATE_KEY`: GPG private key (armor)
- `GPG_PASSPHRASE`: GPG passphrase

**GitHub:**
- `GITHUB_TOKEN`: Automatically provided

### Workflow Steps

1. Checkout code
2. Setup Node.js and Python
3. Install dependencies
4. Setup code signing (platform-specific)
5. Build React app
6. Build Electron app
7. Upload artifacts
8. Generate checksums
9. Create GitHub release (draft)
10. Publish release (if tagged)

### Artifacts

Each build produces:
- Platform installers
- Checksums file
- Auto-update metadata

Download from Actions > [Workflow Run] > Artifacts

## Testing

### Pre-Release Testing

Before releasing:

1. **Build locally** for target platform
2. **Install** the generated installer
3. **Test core features**:
   - Network designer
   - Code editor
   - Shape propagation
   - Debugger
   - AI assistant
   - Project save/load
4. **Test auto-update**:
   - Mock new version
   - Verify download
   - Verify installation
5. **Verify code signing**:
   - Check installer signature
   - Verify no security warnings

### Platform-Specific Testing

**Windows:**
```powershell
# Verify signature
signtool verify /pa "Neural Aquarium Setup.exe"

# Test silent install
"Neural Aquarium Setup.exe" /S

# Test uninstall
"%LOCALAPPDATA%\Programs\neural-aquarium\Uninstall Neural Aquarium.exe" /S
```

**macOS:**
```bash
# Verify signature
codesign --verify --deep --strict --verbose=2 "Neural Aquarium.app"

# Verify notarization
spctl --assess --verbose=4 --type execute "Neural Aquarium.app"

# Check stapling
stapler validate "Neural Aquarium.dmg"
```

**Linux:**
```bash
# Test AppImage
chmod +x Neural-Aquarium-0.3.0-x64.AppImage
./Neural-Aquarium-0.3.0-x64.AppImage

# Install DEB
sudo dpkg -i neural-aquarium_0.3.0_amd64.deb

# Install RPM
sudo rpm -i neural-aquarium-0.3.0.x86_64.rpm
```

## Release Process

Follow [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md) for complete process.

### Quick Release

```bash
# 1. Update version
node build/version-bump.js minor --commit --tag

# 2. Push tag (triggers CI/CD)
git push origin aquarium-v0.3.0

# 3. Monitor GitHub Actions
# Visit: https://github.com/neural-dsl/neural-dsl/actions

# 4. Test draft release
# Download and test installers

# 5. Publish release
# Edit release notes and publish
```

### Version Bumping

```bash
# Patch (0.3.0 -> 0.3.1)
node build/version-bump.js patch

# Minor (0.3.0 -> 0.4.0)
node build/version-bump.js minor

# Major (0.3.0 -> 1.0.0)
node build/version-bump.js major

# Specific version
node build/version-bump.js 1.2.3

# With git commit and tag
node build/version-bump.js minor --commit --tag --push
```

## Troubleshooting

### Build Fails

**"Module not found"**
```bash
npm ci
rm -rf node_modules package-lock.json
npm install
```

**"Rebuild native modules"**
```bash
npm rebuild
```

**Out of memory**
```bash
export NODE_OPTIONS=--max-old-space-size=4096
npm run electron:build
```

### Code Signing Fails

**Windows: "Certificate not found"**
- Verify WIN_CSC_LINK path exists
- Check certificate password
- Ensure certificate is not expired

**macOS: "No identity found"**
- Import certificate to Keychain
- Run `security find-identity -v -p codesigning`
- Unlock keychain if locked

**macOS: "Notarization failed"**
- Check Apple ID credentials
- Verify app-specific password
- Wait 10-15 minutes for Apple servers
- Check email for rejection reasons

### Auto-Update Not Working

- Verify release is published (not draft)
- Check `publish` config in package.json
- Ensure GH_TOKEN has correct permissions
- Check app version < release version
- Look at logs: `~/.config/Neural Aquarium/logs/`

### Platform-Specific Issues

**Windows: Defender flags app**
- Submit to Microsoft for analysis
- Use EV certificate for instant reputation

**macOS: Gatekeeper blocks app**
- Verify app is notarized
- Tell users to right-click > Open first time

**Linux: Missing dependencies**
- Check system requirements
- Install with package manager (handles deps)

## Additional Resources

- [Electron Builder Documentation](https://www.electron.build/)
- [electron-updater Guide](https://www.electron.build/auto-update)
- [CODE_SIGNING.md](./CODE_SIGNING.md)
- [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md)
- [ICONS.md](./ICONS.md)
- [README.md](./README.md)

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting)
2. Search existing GitHub issues
3. Create new issue with:
   - Platform and version
   - Build logs
   - Error messages
   - Steps to reproduce

## License

See LICENSE file in repository root.
