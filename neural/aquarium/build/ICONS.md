# Icon Requirements and Generation

Neural Aquarium requires icons in multiple formats for different platforms.

## Required Icons

### Windows
- `icon.ico` - 256x256 pixels, multi-resolution ICO file
- `file-icon.ico` - 256x256 pixels, for .neural file association

### macOS
- `icon.icns` - Multi-resolution ICNS file containing:
  - 16x16, 32x32, 64x64, 128x128, 256x256, 512x512, 1024x1024
- `background.png` - 540x380 pixels, DMG background image

### Linux
- `icon.png` - 512x512 pixels, main application icon
- `icons/` directory with multiple sizes:
  - 16x16.png
  - 32x32.png
  - 48x48.png
  - 64x64.png
  - 128x128.png
  - 256x256.png
  - 512x512.png

## Design Guidelines

### Application Icon
- Use Neural DSL branding colors
- Should be recognizable at small sizes (16x16)
- Should work on both light and dark backgrounds
- Include some neural network imagery (nodes, connections)
- Keep it simple and clean

### Design Suggestions
- Primary color: Blue (#007ACC) - represents technology and trust
- Accent color: Cyan (#00D4FF) - represents neural connections
- Symbol ideas:
  - Stylized brain/neural network
  - Connected nodes forming "N" or "A"
  - Circuit board pattern
  - Abstract network topology

## Icon Generation

### Option 1: Professional Design
1. Hire a designer or use design tools (Figma, Sketch, Adobe Illustrator)
2. Create master icon at 1024x1024 or vector format
3. Use icon generation tools to create all required formats

### Option 2: Automated Generation

We provide a script to generate icons from a single master image:

```bash
# Install ImageMagick (required)
# macOS
brew install imagemagick

# Ubuntu/Debian
sudo apt-get install imagemagick

# Windows
choco install imagemagick

# Generate all icons from master
node build/generate-icons.js path/to/master-icon.png
```

### Option 3: Use Electron Icon Maker

```bash
npm install -g electron-icon-maker

# Generate from single PNG
electron-icon-maker --input=master-icon.png --output=build
```

### Option 4: Manual Creation

#### Windows ICO
Use online tools or:
```bash
# With ImageMagick
convert master-icon.png -define icon:auto-resize=256,128,96,64,48,32,16 icon.ico
```

#### macOS ICNS
```bash
# Create iconset directory
mkdir icon.iconset

# Generate all sizes
sips -z 16 16     master-icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32     master-icon.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     master-icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64     master-icon.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   master-icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256   master-icon.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   master-icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512   master-icon.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   master-icon.png --out icon.iconset/icon_512x512.png
sips -z 1024 1024 master-icon.png --out icon.iconset/icon_512x512@2x.png

# Create ICNS
iconutil -c icns icon.iconset
```

#### Linux Icons
```bash
mkdir -p icons
for size in 16 32 48 64 128 256 512; do
  convert master-icon.png -resize ${size}x${size} icons/${size}x${size}.png
done
```

## DMG Background

The DMG background image should:
- Be 540x380 pixels
- Have a subtle, professional look
- Match brand colors
- Include subtle instructions ("Drag to Applications")
- Use semi-transparent elements

Example creation with ImageMagick:
```bash
convert -size 540x380 xc:'#1e1e1e' \
  -font Arial -pointsize 24 -fill white \
  -annotate +170+200 'Neural Aquarium' \
  -font Arial -pointsize 14 -fill '#aaaaaa' \
  -annotate +140+230 'Drag the icon to Applications folder' \
  background.png
```

## Placeholder Icons

For development, you can use simple placeholder icons:

```bash
# Create simple placeholder
convert -size 512x512 xc:'#007ACC' \
  -font Arial-Bold -pointsize 200 -fill white \
  -gravity center -annotate +0+0 'NA' \
  master-icon.png
```

## Icon Testing

After generating icons:

1. **Windows**: Check ICO has multiple resolutions
   ```cmd
   magick identify icon.ico
   ```

2. **macOS**: Verify ICNS structure
   ```bash
   iconutil -c iconset icon.icns
   ls icon.iconset/
   ```

3. **Linux**: Check all sizes exist
   ```bash
   ls -lh icons/
   ```

4. **Visual Testing**:
   - Build and install the app
   - Check desktop shortcut icon
   - Check taskbar/dock icon
   - Check file association icon (for .neural files)
   - Test on light and dark system themes

## Resources

- [Electron Builder Icons](https://www.electron.build/icons)
- [Icon Generator Tools](https://www.electron.build/icons#icon-generator-tools)
- [Apple Human Interface Guidelines - Icons](https://developer.apple.com/design/human-interface-guidelines/macos/icons-and-images/app-icon/)
- [Windows App Icon Construction](https://docs.microsoft.com/en-us/windows/apps/design/style/iconography/app-icon-construction)
- [Freedesktop Icon Theme Specification](https://specifications.freedesktop.org/icon-theme-spec/icon-theme-spec-latest.html)

## Current Status

**TODO**: Replace placeholder icons with proper branded icons before release.

The build system will work without icons, but installers will use default Electron icons. For a professional release, custom icons are essential.
