# Testing Guide for Packaged Application

This guide covers testing the packaged Neural Aquarium application before release.

## Test Categories

### 1. Installation Testing

#### Windows
- [ ] NSIS installer launches without errors
- [ ] Installation directory can be customized
- [ ] Desktop shortcut is created
- [ ] Start Menu entry is created
- [ ] Application launches from shortcut
- [ ] MSI installer works
- [ ] MSI supports per-user and per-machine installation
- [ ] Uninstaller removes all files and registry entries
- [ ] Reinstallation over existing installation works

#### macOS
- [ ] DMG opens and displays correctly
- [ ] Drag-to-Applications works
- [ ] App launches from Applications folder
- [ ] First launch doesn't show security warnings (if notarized)
- [ ] Right-click > Open works for unsigned/unnotarized builds
- [ ] App icon appears correctly in Dock
- [ ] ZIP archive extracts correctly

#### Linux
- [ ] AppImage has execute permissions
- [ ] AppImage runs without installation
- [ ] AppImage integrates with desktop environment
- [ ] DEB package installs with dpkg
- [ ] DEB package installs with apt
- [ ] Desktop entry appears in application menu
- [ ] RPM package installs with rpm
- [ ] RPM package installs with dnf/yum
- [ ] Uninstallation removes all files

### 2. Code Signing Verification

#### Windows
```powershell
# Verify digital signature
signtool verify /pa /v "Neural Aquarium Setup.exe"

# Expected: "Successfully verified"
```

#### macOS
```bash
# Verify code signature
codesign --verify --deep --strict --verbose=2 "Neural Aquarium.app"

# Verify notarization
spctl --assess --verbose=4 --type execute "Neural Aquarium.app"

# Check stapling
stapler validate "Neural Aquarium.dmg"

# Expected: All commands succeed without errors
```

#### Linux
```bash
# Verify DEB signature
dpkg-sig --verify neural-aquarium_0.3.0_amd64.deb

# Verify RPM signature
rpm --checksig neural-aquarium-0.3.0.x86_64.rpm
```

### 3. Application Launch

- [ ] Application starts within 5 seconds
- [ ] Splash screen displays (if implemented)
- [ ] Main window opens at correct size
- [ ] Window is centered on screen
- [ ] Application icon displays in taskbar/dock
- [ ] Menu bar loads correctly
- [ ] No console errors in DevTools (Help > Toggle DevTools)

### 4. Core Functionality

#### Network Designer
- [ ] Layer palette loads with all layer types
- [ ] Drag-and-drop works from palette to canvas
- [ ] Layers can be connected
- [ ] Layer properties can be edited
- [ ] Zoom in/out works
- [ ] Pan with mouse works
- [ ] Selection works
- [ ] Delete layer works
- [ ] Undo/Redo works
- [ ] Export to DSL code works

#### Code Editor
- [ ] Syntax highlighting works
- [ ] Autocomplete suggestions appear
- [ ] Error diagnostics show inline
- [ ] Line numbers display
- [ ] Find/Replace works
- [ ] Format document works
- [ ] Save file works (Ctrl/Cmd+S)
- [ ] Monaco editor loads without errors

#### Shape Propagation
- [ ] Shape inference runs automatically
- [ ] Shape errors are highlighted
- [ ] Shape visualization updates
- [ ] Shape details show in properties panel
- [ ] Shape propagation works with complex networks

#### Debugger
- [ ] Breakpoints can be set
- [ ] Execution pauses at breakpoints
- [ ] Step Over works
- [ ] Step Into works
- [ ] Step Out works
- [ ] Continue works
- [ ] Variable inspection shows values
- [ ] Watch expressions work
- [ ] Call stack displays
- [ ] Execution timeline updates

#### AI Assistant
- [ ] Chat interface loads
- [ ] Can send messages
- [ ] AI responses appear
- [ ] Code suggestions can be inserted
- [ ] Language selection works
- [ ] Chat history persists

#### Project Management
- [ ] New project creates correctly
- [ ] Open project works
- [ ] Save project works
- [ ] Save As works
- [ ] Recent projects list updates
- [ ] Project files (.neural) open correctly
- [ ] File tree displays correctly

### 5. Menu and Shortcuts

#### File Menu
- [ ] New Project (Ctrl/Cmd+N)
- [ ] Open Project (Ctrl/Cmd+O)
- [ ] Save (Ctrl/Cmd+S)
- [ ] Save As (Ctrl/Cmd+Shift+S)
- [ ] Exit (Ctrl/Cmd+Q)

#### Edit Menu
- [ ] Undo (Ctrl/Cmd+Z)
- [ ] Redo (Ctrl/Cmd+Shift+Z)
- [ ] Cut (Ctrl/Cmd+X)
- [ ] Copy (Ctrl/Cmd+C)
- [ ] Paste (Ctrl/Cmd+V)
- [ ] Select All (Ctrl/Cmd+A)

#### View Menu
- [ ] Reload (Ctrl/Cmd+R)
- [ ] Force Reload (Ctrl/Cmd+Shift+R)
- [ ] Toggle DevTools (Ctrl/Cmd+Shift+I)
- [ ] Reset Zoom (Ctrl/Cmd+0)
- [ ] Zoom In (Ctrl/Cmd++)
- [ ] Zoom Out (Ctrl/Cmd+-)
- [ ] Toggle Fullscreen (F11)

#### Help Menu
- [ ] Documentation (opens browser)
- [ ] Check for Updates
- [ ] About (shows version info)

### 6. Auto-Update Testing

#### Update Check
- [ ] Check for updates from Help menu
- [ ] Update notification appears if available
- [ ] No update notification if on latest version
- [ ] Update check on startup works

#### Update Download
- [ ] Download progress shows percentage
- [ ] Download speed is displayed
- [ ] Download can be cancelled (if implemented)
- [ ] Download completes successfully

#### Update Installation
- [ ] Installation prompt appears
- [ ] "Install Now" restarts and updates
- [ ] "Install Later" postpones update
- [ ] App restarts correctly
- [ ] New version runs after update
- [ ] Settings/data persist after update

#### Update Rollback (if implemented)
- [ ] Failed update rolls back to previous version
- [ ] Error message is shown
- [ ] App remains functional

### 7. File Associations

#### Windows
- [ ] .neural files show Neural Aquarium icon
- [ ] Double-click opens file in Neural Aquarium
- [ ] Right-click > Open With shows Neural Aquarium

#### macOS
- [ ] .neural files show Neural Aquarium icon
- [ ] Double-click opens file in Neural Aquarium
- [ ] Get Info shows Neural Aquarium as default app

#### Linux
- [ ] .neural files show Neural Aquarium icon (if desktop integrated)
- [ ] Double-click opens file in Neural Aquarium
- [ ] MIME type association works

### 8. Performance Testing

- [ ] Startup time < 5 seconds
- [ ] Memory usage < 500MB at idle
- [ ] Memory usage doesn't grow over time (no leaks)
- [ ] CPU usage < 5% at idle
- [ ] Large network (100+ layers) loads in < 10 seconds
- [ ] Shape propagation on large network < 2 seconds
- [ ] UI remains responsive during computation
- [ ] Multiple projects can be opened

### 9. Settings Persistence

- [ ] Window size/position persists
- [ ] Theme selection persists
- [ ] Recent projects persist
- [ ] Editor preferences persist
- [ ] Debugger breakpoints persist

### 10. Error Handling

- [ ] Invalid .neural file shows error message
- [ ] Missing dependencies show helpful error
- [ ] Network errors show retry option
- [ ] Python backend failure shows error
- [ ] App doesn't crash on errors
- [ ] Error logs are written to disk

### 11. Localization (if implemented)

- [ ] Language selection works
- [ ] UI text translates correctly
- [ ] Date/time formats are locale-appropriate
- [ ] Number formats are locale-appropriate

### 12. Accessibility (if implemented)

- [ ] Keyboard navigation works throughout app
- [ ] Tab order is logical
- [ ] Focus indicators are visible
- [ ] Screen reader announcements work
- [ ] High contrast mode works

### 13. Security

- [ ] No console warnings about mixed content
- [ ] No console warnings about CSP violations
- [ ] External links open in browser, not in-app
- [ ] No eval() usage in production
- [ ] Node integration is disabled in renderer

### 14. Cross-Platform Consistency

Compare behavior across platforms:
- [ ] UI layout is consistent
- [ ] Features work identically
- [ ] Keyboard shortcuts are adapted (Ctrl vs Cmd)
- [ ] File paths work correctly
- [ ] Line endings don't cause issues

### 15. Upgrade Testing

If updating from previous version:
- [ ] Settings migrate correctly
- [ ] Projects open without errors
- [ ] Breakpoints are preserved
- [ ] Recent projects list is maintained
- [ ] No data loss

### 16. Uninstallation

#### Windows
- [ ] Uninstaller runs from Control Panel
- [ ] All files are removed
- [ ] Registry entries are cleaned
- [ ] Start Menu entry is removed
- [ ] Desktop shortcut is removed
- [ ] User data is preserved (or prompt to delete)

#### macOS
- [ ] Drag to Trash works
- [ ] No leftover files in /Applications
- [ ] User data in ~/Library is preserved

#### Linux
- [ ] Package manager uninstall works
- [ ] Desktop integration is removed
- [ ] User data in ~/.config is preserved

## Automated Testing

For regression testing, consider:

```bash
# Run unit tests
npm test

# Run integration tests
npm run test:integration

# Run E2E tests
npm run test:e2e

# Generate coverage report
npm run test:coverage
```

## Test Environments

### Minimum Requirements
- **Windows**: Windows 10 (1809+)
- **macOS**: macOS 10.13 High Sierra
- **Linux**: Ubuntu 18.04, Fedora 32

### Recommended Test Platforms
- **Windows**: 10, 11 (both x64 and ARM64 if possible)
- **macOS**: Intel and Apple Silicon
- **Linux**: Ubuntu 20.04, Ubuntu 22.04, Fedora 36+

## Test Data

Use these test cases:

1. **Simple Network**: `examples/mnist_cnn.neural`
2. **Complex Network**: Create 100+ layer network
3. **Invalid File**: Malformed .neural file
4. **Large File**: Network with MB+ of parameters
5. **Edge Cases**: Empty network, single layer, disconnected layers

## Bug Reporting Template

When filing bugs found during testing:

```markdown
**Platform**: Windows 11 x64
**Version**: 0.3.0
**Installer**: NSIS

**Steps to Reproduce**:
1. Open application
2. Create new project
3. Click "Add Layer"

**Expected**: Layer palette should open

**Actual**: Application crashes

**Logs**: (attach logs from ~/.config/Neural Aquarium/logs/)

**Screenshots**: (attach if applicable)
```

## Sign-Off Checklist

Before release, ensure:
- [ ] All critical tests pass
- [ ] No P0/P1 bugs remain
- [ ] Code signing verified on all platforms
- [ ] Auto-update tested end-to-end
- [ ] Performance is acceptable
- [ ] Documentation is updated
- [ ] Release notes are written

## Post-Release Monitoring

After release:
- Monitor GitHub Issues for bug reports
- Check analytics for crash rates (if implemented)
- Monitor auto-update server logs
- Review user feedback on forums/social media
- Track download numbers
- Prepare hotfix if critical issues found
