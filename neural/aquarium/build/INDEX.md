# Build System Documentation Index

This directory contains all files and documentation for building, packaging, and distributing Neural Aquarium as a desktop application.

## Quick Navigation

- **Getting Started**: [PACKAGING_GUIDE.md](./PACKAGING_GUIDE.md)
- **Code Signing**: [CODE_SIGNING.md](./CODE_SIGNING.md)
- **Release Process**: [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md)
- **Testing**: [TESTING.md](./TESTING.md)
- **Icons**: [ICONS.md](./ICONS.md)
- **Build System**: [README.md](./README.md)

## Documentation Files

### Primary Guides

| File | Description | Audience |
|------|-------------|----------|
| [PACKAGING_GUIDE.md](./PACKAGING_GUIDE.md) | Complete packaging and distribution guide | All developers |
| [CODE_SIGNING.md](./CODE_SIGNING.md) | Code signing setup for all platforms | Release managers |
| [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md) | Step-by-step release process | Release managers |
| [README.md](./README.md) | Build system overview | All developers |

### Specialized Guides

| File | Description | Audience |
|------|-------------|----------|
| [TESTING.md](./TESTING.md) | Testing checklist for releases | QA, Release managers |
| [ICONS.md](./ICONS.md) | Icon requirements and generation | Designers, Developers |
| [INDEX.md](./INDEX.md) | This file - documentation index | All |

## Configuration Files

### Electron Builder

| File | Description |
|------|-------------|
| `electron-builder.yml` | Main Electron Builder configuration |
| `dev-app-update.yml` | Development auto-update configuration |

### macOS

| File | Description |
|------|-------------|
| `entitlements.mac.plist` | macOS app entitlements for sandboxing |
| `notarize.js` | macOS notarization script |

### Linux

| File | Description |
|------|-------------|
| `linux/after-install.sh` | Post-installation script (deb/rpm) |
| `linux/after-remove.sh` | Post-removal script (deb/rpm) |

## Scripts

| File | Description | Usage |
|------|-------------|-------|
| `build.sh` | Bash build script | `./build.sh -p linux -s -r` |
| `build.ps1` | PowerShell build script | `.\build.ps1 -Platform win -Sign` |
| `version-bump.js` | Version management utility | `node version-bump.js minor` |
| `generate-icons.js` | Icon generation from master image | `node generate-icons.js icon.png` |

## Assets

### Icons (to be created)

| File | Platform | Description |
|------|----------|-------------|
| `icon.ico` | Windows | Application icon (multi-resolution) |
| `file-icon.ico` | Windows | .neural file icon |
| `icon.icns` | macOS | Application icon bundle |
| `icon.png` | Linux | Application icon (512x512) |
| `icons/*.png` | Linux | Icon sizes (16-512px) |
| `background.png` | macOS | DMG background image |

See [ICONS.md](./ICONS.md) for generation instructions.

## Directory Structure

```
build/
├── INDEX.md                    # This file
├── README.md                   # Build system overview
├── PACKAGING_GUIDE.md          # Complete packaging guide
├── CODE_SIGNING.md             # Code signing guide
├── RELEASE_CHECKLIST.md        # Release process
├── TESTING.md                  # Testing checklist
├── ICONS.md                    # Icon requirements
├── .gitignore                  # Build artifacts to ignore
│
├── electron-builder.yml        # Main builder config
├── dev-app-update.yml          # Dev update config
├── entitlements.mac.plist      # macOS entitlements
├── notarize.js                 # macOS notarization
│
├── build.sh                    # Bash build script
├── build.ps1                   # PowerShell build script
├── version-bump.js             # Version management
├── generate-icons.js           # Icon generation
│
├── icon.ico                    # Windows app icon (TODO)
├── file-icon.ico               # Windows file icon (TODO)
├── icon.icns                   # macOS app icon (TODO)
├── icon.png                    # Linux app icon (TODO)
├── background.png              # DMG background (TODO)
│
├── icons/                      # Linux icon sizes (TODO)
│   ├── 16x16.png
│   ├── 32x32.png
│   ├── 48x48.png
│   ├── 64x64.png
│   ├── 128x128.png
│   ├── 256x256.png
│   └── 512x512.png
│
└── linux/                      # Linux scripts
    ├── after-install.sh
    └── after-remove.sh
```

## Common Tasks

### Building

```bash
# Quick development build
npm run electron:build

# Full release build (all platforms)
npm run electron:build:all

# Platform-specific
npm run electron:build:win
npm run electron:build:mac
npm run electron:build:linux

# Using build scripts
./build/build.sh --platform all --sign --release
.\build\build.ps1 -Platform win -Sign -Release
```

### Version Management

```bash
# Bump version
node build/version-bump.js patch    # 0.3.0 -> 0.3.1
node build/version-bump.js minor    # 0.3.0 -> 0.4.0
node build/version-bump.js major    # 0.3.0 -> 1.0.0

# With git operations
node build/version-bump.js minor --commit --tag --push
```

### Icon Generation

```bash
# Generate all icons from master
node build/generate-icons.js path/to/master-icon.png

# Generate placeholder for development
node build/generate-icons.js
```

### Code Signing Setup

See [CODE_SIGNING.md](./CODE_SIGNING.md) for detailed instructions.

**Quick setup:**
```bash
# Windows
export WIN_CSC_LINK=/path/to/cert.pfx
export WIN_CSC_KEY_PASSWORD=password

# macOS
export APPLE_ID=your@email.com
export APPLE_ID_PASSWORD=xxxx-xxxx-xxxx-xxxx
export APPLE_TEAM_ID=TEAM_ID

# Linux
export GPG_PASSPHRASE=passphrase
```

### Testing

See [TESTING.md](./TESTING.md) for complete checklist.

**Quick tests:**
```bash
# Build and test locally
npm run electron:build
# Install and manually test

# Verify code signing
# Windows
signtool verify /pa Neural-Aquarium.exe

# macOS
codesign --verify --deep --strict Neural-Aquarium.app
spctl --assess --verbose Neural-Aquarium.app

# Linux
dpkg-sig --verify neural-aquarium.deb
```

### Releasing

See [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md) for complete process.

**Quick release:**
```bash
# 1. Bump version
node build/version-bump.js minor --commit --tag

# 2. Push tag (triggers CI/CD)
git push origin aquarium-v0.4.0

# 3. Monitor GitHub Actions
# 4. Test draft release
# 5. Publish release
```

## CI/CD

GitHub Actions workflow: `.github/workflows/aquarium-release.yml`

**Triggers:**
- Push tag matching `aquarium-v*.*.*`
- Manual workflow dispatch

**Builds:**
- Windows (NSIS, MSI)
- macOS (DMG, ZIP)
- Linux (AppImage, DEB, RPM)

**Outputs:**
- Draft GitHub Release
- Signed installers
- Checksums
- Auto-update metadata

## Environment Variables

### Development

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment mode | `production` |
| `ELECTRON_UPDATE_URL` | Custom update server | GitHub Releases |
| `CSC_IDENTITY_AUTO_DISCOVERY` | Auto-find certificates | `true` |

### Windows Code Signing

| Variable | Description |
|----------|-------------|
| `WIN_CSC_LINK` | Certificate file path |
| `WIN_CSC_KEY_PASSWORD` | Certificate password |

### macOS Code Signing

| Variable | Description |
|----------|-------------|
| `APPLE_ID` | Apple ID email |
| `APPLE_ID_PASSWORD` | App-specific password |
| `APPLE_TEAM_ID` | Team ID |
| `CSC_LINK` | P12 certificate path |
| `CSC_KEY_PASSWORD` | Certificate password |

### Linux Signing

| Variable | Description |
|----------|-------------|
| `GPG_PASSPHRASE` | GPG key passphrase |

### Publishing

| Variable | Description |
|----------|-------------|
| `GH_TOKEN` | GitHub personal access token |
| `PUBLISH` | Publish mode: `always`, `never` |

## Troubleshooting

### Common Issues

**Build fails with "Module not found"**
```bash
npm ci
rm -rf node_modules
npm install
```

**Code signing fails**
- Check certificate validity and expiration
- Verify environment variables are set
- Check certificate password is correct

**Auto-update not working**
- Ensure release is published (not draft)
- Verify `publish` config in package.json
- Check GH_TOKEN permissions
- Review app logs

See [PACKAGING_GUIDE.md](./PACKAGING_GUIDE.md#troubleshooting) for more.

## Getting Help

1. **Check Documentation**: Start with relevant guide above
2. **Search Issues**: [GitHub Issues](https://github.com/neural-dsl/neural-dsl/issues)
3. **Ask for Help**: Create new issue with details
4. **Community**: Discussions, Discord, etc.

## Contributing

When updating build system:

1. Test changes on all platforms
2. Update relevant documentation
3. Add to [CHANGELOG.md](../CHANGELOG.md)
4. Create pull request

## Related Files

Outside this directory:

| File | Description |
|------|-------------|
| `../package.json` | Main package.json with build config |
| `../electron/` | Electron main process code |
| `.github/workflows/aquarium-release.yml` | CI/CD workflow |
| `../CHANGELOG.md` | Version history |
| `../LICENSE` | License file |

## Resources

### Electron Builder
- [Documentation](https://www.electron.build/)
- [Configuration](https://www.electron.build/configuration/configuration)
- [Code Signing](https://www.electron.build/code-signing)
- [Auto Update](https://www.electron.build/auto-update)

### Platform Guidelines
- [Windows Packaging](https://docs.microsoft.com/en-us/windows/apps/desktop/modernize/desktop-to-uwp-packaging-dot-net)
- [macOS App Distribution](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Linux Packaging](https://www.electron.build/configuration/linux)

### Tools
- [ImageMagick](https://imagemagick.org/) - Icon generation
- [iconutil](https://developer.apple.com/library/archive/documentation/GraphicsAnimation/Conceptual/HighResolutionOSX/Optimizing/Optimizing.html) - macOS icon creation
- [signtool](https://docs.microsoft.com/en-us/windows/win32/seccrypto/signtool) - Windows signing

## License

See [LICENSE](../../LICENSE) in repository root.

---

**Last Updated**: December 2024
**Maintainer**: Neural DSL Team
**Version**: 0.3.0
