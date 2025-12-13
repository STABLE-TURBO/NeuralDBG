# Neural Aquarium Packaging - Quick Start Guide

Get up and running with packaging Neural Aquarium in 5 minutes.

## Prerequisites

- Node.js 18+ and npm
- Git
- Platform-specific requirements (see below)

## Installation

```bash
cd neural/aquarium
npm install
```

## Quick Commands

### Development

```bash
# Run in development mode with hot reload
npm run electron:dev
```

### Building

```bash
# Build for your current platform
npm run electron:build

# Platform-specific builds
npm run electron:build:win     # Windows
npm run electron:build:mac     # macOS
npm run electron:build:linux   # Linux

# Build for all platforms (requires appropriate OS)
npm run electron:build:all
```

### Output

Built installers will be in `dist/` directory.

## Platform-Specific Setup

### Windows

**Requirements:**
- Windows 10+
- Windows SDK (for signing)

**Optional: Code Signing**
```powershell
$env:WIN_CSC_LINK = "C:\path\to\certificate.pfx"
$env:WIN_CSC_KEY_PASSWORD = "your-password"
```

### macOS

**Requirements:**
- macOS 10.13+
- Xcode Command Line Tools: `xcode-select --install`

**Optional: Code Signing**
```bash
export APPLE_ID="your@email.com"
export APPLE_ID_PASSWORD="xxxx-xxxx-xxxx-xxxx"  # App-specific password
export APPLE_TEAM_ID="YOUR_TEAM_ID"
```

### Linux

**Requirements:**
- Ubuntu 18.04+ or equivalent
- Additional packages:
```bash
sudo apt-get install -y rpm fakeroot dpkg
```

## Skip Code Signing (Development)

```bash
export CSC_IDENTITY_AUTO_DISCOVERY=false
npm run electron:build
```

## Testing the Build

### Windows
```powershell
# Install
.\dist\Neural-Aquarium-Setup-0.3.0.exe

# Verify signature
signtool verify /pa ".\dist\Neural-Aquarium-Setup-0.3.0.exe"
```

### macOS
```bash
# Open DMG
open dist/Neural-Aquarium-0.3.0.dmg

# Verify signature
codesign --verify --deep --strict "Neural Aquarium.app"
```

### Linux
```bash
# Run AppImage
chmod +x dist/Neural-Aquarium-0.3.0-x86_64.AppImage
./dist/Neural-Aquarium-0.3.0-x86_64.AppImage

# Install DEB
sudo dpkg -i dist/neural-aquarium_0.3.0_amd64.deb

# Install RPM
sudo rpm -i dist/neural-aquarium-0.3.0.x86_64.rpm
```

## Creating a Release

### 1. Bump Version

```bash
# Patch: 0.3.0 → 0.3.1
node build/version-bump.js patch

# Minor: 0.3.0 → 0.4.0
node build/version-bump.js minor

# Major: 0.3.0 → 1.0.0
node build/version-bump.js major
```

### 2. Commit and Tag

```bash
git add .
git commit -m "Release v0.4.0"
git tag aquarium-v0.4.0
git push origin main --tags
```

### 3. GitHub Actions Will:
- Build for Windows, macOS, Linux
- Sign installers (if configured)
- Create draft GitHub Release
- Upload installers and checksums

### 4. Publish Release
- Go to GitHub Releases
- Review draft release
- Edit release notes if needed
- Click "Publish release"

## Troubleshooting

### "Module not found"
```bash
rm -rf node_modules package-lock.json
npm install
```

### "Cannot find electron"
```bash
npm install electron --save-dev
```

### Build fails
```bash
# Clean and rebuild
rm -rf dist node_modules
npm install
npm run electron:build
```

### Code signing fails
- Check certificate is valid and not expired
- Verify environment variables are set correctly
- See `build/CODE_SIGNING.md` for detailed setup

## Common Issues

**Windows: "Certificate not found"**
- Check `WIN_CSC_LINK` path exists
- Verify certificate password

**macOS: "No identity found"**
- Import certificate to Keychain
- Check with: `security find-identity -v -p codesigning`

**Linux: "Missing dependencies"**
- Install build tools: `sudo apt-get install rpm fakeroot dpkg`

## File Locations

- **Configuration**: `build/electron-builder.yml`
- **Main Process**: `electron/main.js`
- **Preload Script**: `electron/preload.js`
- **Auto-Update UI**: `src/components/UpdateNotification.tsx`
- **Build Scripts**: `build/build.sh`, `build/build.ps1`
- **CI/CD Workflow**: `.github/workflows/aquarium-release.yml`

## Documentation

Full documentation in `build/` directory:

- **PACKAGING_GUIDE.md** - Complete guide
- **CODE_SIGNING.md** - Code signing setup
- **RELEASE_CHECKLIST.md** - Release process
- **TESTING.md** - Testing guide
- **README.md** - Build system overview

## Getting Help

1. Check documentation in `build/` directory
2. Search [GitHub Issues](https://github.com/neural-dsl/neural-dsl/issues)
3. Create new issue with platform, version, and error details

## Next Steps

After your first build:

1. ✅ Test the installer on target platform
2. ✅ Set up code signing certificates (see `CODE_SIGNING.md`)
3. ✅ Configure GitHub Actions secrets
4. ✅ Read the release checklist (`RELEASE_CHECKLIST.md`)
5. ✅ Create professional icons (see `ICONS.md`)
6. ✅ Test auto-update functionality

## Resources

- [Electron Builder Docs](https://www.electron.build/)
- [electron-updater Guide](https://www.electron.build/auto-update)
- [Code Signing Guide](build/CODE_SIGNING.md)
- [Release Checklist](build/RELEASE_CHECKLIST.md)

---

**Quick Help:**
- Build: `npm run electron:build`
- Dev: `npm run electron:dev`
- Release: Tag with `aquarium-v*` and push
- Docs: See `build/` directory
