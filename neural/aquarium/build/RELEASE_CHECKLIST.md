# Release Checklist

Use this checklist when preparing a new release of Neural Aquarium.

## Pre-Release (1 Week Before)

- [ ] Review and merge all pending pull requests
- [ ] Update dependencies to latest stable versions
- [ ] Run full test suite and fix any failures
- [ ] Test on all target platforms (Windows, macOS, Linux)
- [ ] Review and close related GitHub issues
- [ ] Update documentation for new features
- [ ] Check that all example files work correctly
- [ ] Verify Python backend compatibility

## Code Freeze (3 Days Before)

- [ ] Create release branch: `git checkout -b release/v0.x.0`
- [ ] Update version in `package.json`
- [ ] Update version in `neural/aquarium/__init__.py` (if applicable)
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Update version references in documentation
- [ ] Run linters and fix all issues
- [ ] Build and test installers locally on all platforms

## Code Signing Setup

### Windows
- [ ] Verify code signing certificate is valid and not expired
- [ ] Test signing process with test certificate
- [ ] Ensure `WIN_CSC_LINK` and `WIN_CSC_KEY_PASSWORD` secrets are set
- [ ] Verify timestamp server is accessible

### macOS
- [ ] Verify Developer ID certificate is valid
- [ ] Test notarization process
- [ ] Ensure `APPLE_ID`, `APPLE_ID_PASSWORD`, `APPLE_TEAM_ID` secrets are set
- [ ] Verify Xcode Command Line Tools are installed on build machine
- [ ] Test stapling process

### Linux
- [ ] Verify GPG key is available and valid
- [ ] Test package signing
- [ ] Ensure `GPG_PRIVATE_KEY` and `GPG_PASSPHRASE` secrets are set
- [ ] Update public key in distribution channels

## Build Verification

### Local Builds

#### Windows
- [ ] Build NSIS installer: `npm run electron:build:win`
- [ ] Install and test NSIS installer
- [ ] Verify desktop shortcut creation
- [ ] Verify Start Menu entry
- [ ] Test uninstaller
- [ ] Build MSI installer
- [ ] Install and test MSI installer
- [ ] Verify code signature: `signtool verify /pa Neural-Aquarium-*.exe`

#### macOS
- [ ] Build DMG: `npm run electron:build:mac`
- [ ] Install from DMG
- [ ] Verify app signature: `codesign --verify --deep --strict Neural\ Aquarium.app`
- [ ] Verify notarization: `spctl --assess --verbose Neural\ Aquarium.app`
- [ ] Test on both Intel and Apple Silicon (if possible)
- [ ] Verify DMG appearance and background

#### Linux
- [ ] Build AppImage: `npm run electron:build:linux`
- [ ] Test AppImage on Ubuntu
- [ ] Build and test .deb package on Debian/Ubuntu
- [ ] Build and test .rpm package on Fedora/RHEL
- [ ] Verify desktop integration
- [ ] Verify MIME type association (.neural files)
- [ ] Test both x64 and arm64 builds (if possible)

### Feature Testing
- [ ] Test network designer drag-and-drop
- [ ] Test code editor syntax highlighting
- [ ] Test shape propagation visualization
- [ ] Test debugger (breakpoints, step through, variable inspection)
- [ ] Test AI assistant integration
- [ ] Test project save/load
- [ ] Test export to TensorFlow/PyTorch/ONNX
- [ ] Test example networks load correctly
- [ ] Test settings persistence

### Performance Testing
- [ ] Test with large network (100+ layers)
- [ ] Test memory usage over extended session
- [ ] Test startup time
- [ ] Test file loading performance
- [ ] Test shape propagation speed

### Auto-Update Testing
- [ ] Test update check on startup
- [ ] Test update download progress
- [ ] Test update installation
- [ ] Verify rollback on failed update
- [ ] Test "skip this version" functionality

## CI/CD Verification

- [ ] Verify GitHub Actions workflow is up to date
- [ ] Check all required secrets are set in repository settings:
  - `WIN_CSC_LINK`, `WIN_CSC_KEY_PASSWORD`
  - `APPLE_ID`, `APPLE_ID_PASSWORD`, `APPLE_TEAM_ID`
  - `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`
  - `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`
  - `GH_TOKEN` (automatic, but verify permissions)
- [ ] Test workflow with manual trigger before tagging
- [ ] Verify artifact upload works correctly
- [ ] Verify checksum generation

## Release Process

### Tag and Push
```bash
# Ensure you're on the release branch
git checkout release/v0.x.0

# Create and push tag
git tag -a aquarium-v0.x.0 -m "Release version 0.x.0"
git push origin aquarium-v0.x.0
```

### Monitor CI/CD
- [ ] Watch GitHub Actions workflow progress
- [ ] Verify builds complete successfully on all platforms
- [ ] Check build logs for warnings or errors
- [ ] Verify all artifacts are uploaded
- [ ] Verify checksums are generated

### Verify Release
- [ ] Check draft release is created in GitHub
- [ ] Verify all installers are attached to release
- [ ] Verify checksums are attached
- [ ] Review auto-generated release notes
- [ ] Edit release notes for clarity and completeness
- [ ] Add upgrade instructions if breaking changes
- [ ] Add screenshots if UI changes

### Test Release Assets
- [ ] Download and test Windows installer from GitHub
- [ ] Download and test macOS DMG from GitHub
- [ ] Download and test Linux AppImage from GitHub
- [ ] Verify checksums match
- [ ] Test auto-update from previous version

## Publish Release

- [ ] Change release from "Draft" to "Published"
- [ ] Merge release branch to main: `git merge release/v0.x.0`
- [ ] Tag main branch: `git tag -f aquarium-v0.x.0 && git push --tags`
- [ ] Delete release branch: `git branch -d release/v0.x.0`
- [ ] Update Homebrew cask (if applicable)
- [ ] Update Chocolatey package (if applicable)
- [ ] Update Snapcraft (if applicable)

## Post-Release

### Announcements
- [ ] Post release announcement on GitHub Discussions
- [ ] Update project README with latest version
- [ ] Tweet about release (if applicable)
- [ ] Post on relevant forums/communities
- [ ] Update documentation website
- [ ] Send newsletter to users (if applicable)

### Monitoring
- [ ] Monitor GitHub Issues for release-related bugs
- [ ] Check download statistics
- [ ] Monitor auto-update server logs
- [ ] Watch for code signing issues
- [ ] Check crash reports (if telemetry enabled)

### Version Bump
- [ ] Increment version in `package.json` to next development version
- [ ] Add "Unreleased" section to `CHANGELOG.md`
- [ ] Commit version bump: `git commit -am "Bump version to 0.x.1-dev"`

## Hotfix Process (If Critical Bug Found)

1. Create hotfix branch: `git checkout -b hotfix/v0.x.1`
2. Fix the bug
3. Update version: `0.x.0` → `0.x.1`
4. Update CHANGELOG.md
5. Test fix thoroughly
6. Tag and release: `aquarium-v0.x.1`
7. Merge to main and develop
8. Monitor auto-updates

## Rollback Procedure (If Release is Broken)

1. Mark GitHub release as pre-release or delete it
2. This prevents auto-updates from downloading broken version
3. Fix issues in new hotfix release
4. Users on broken version will auto-update to fix

## Platform-Specific Notes

### Windows
- MSI installers require elevation
- NSIS installers support per-user installation
- Test on Windows 10 and Windows 11
- Verify Windows Defender doesn't flag app

### macOS
- First launch requires right-click → Open (Gatekeeper)
- Test on both Intel and Apple Silicon
- Verify app works on latest macOS version
- Check that app doesn't trigger security warnings

### Linux
- AppImage should work on any recent distribution
- .deb packages tested on Ubuntu LTS versions
- .rpm packages tested on Fedora
- Verify icon appears in application menu

## Useful Commands

```bash
# Check certificate expiration (macOS)
security find-identity -v -p codesigning

# Verify Windows signature
signtool verify /pa /v Neural-Aquarium-0.3.0-win-x64.exe

# Verify macOS signature
codesign --verify --deep --strict --verbose=2 "Neural Aquarium.app"

# Verify macOS notarization
spctl --assess --verbose=4 --type execute "Neural Aquarium.app"

# Check DMG stapling
stapler validate Neural-Aquarium-0.3.0-mac-universal.dmg

# Verify checksums
sha256sum -c checksums-linux.txt
shasum -a 256 -c checksums-macos.txt
Get-FileHash Neural-Aquarium-0.3.0-win-x64.exe -Algorithm SHA256

# Test auto-update locally
export ELECTRON_UPDATE_URL=http://localhost:8080/releases
python -m http.server 8080 --directory dist/
```

## Emergency Contacts

- **Code Signing Issues**: [Contact information]
- **CI/CD Pipeline**: [Contact information]
- **Release Manager**: [Contact information]

## Post-Mortem

After each release, document:
- What went well
- What could be improved
- Issues encountered and how they were resolved
- Time spent on each phase
- Update this checklist based on learnings
