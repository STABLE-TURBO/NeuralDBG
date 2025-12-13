#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const buildDir = __dirname;
const iconsDir = path.join(buildDir, 'icons');

function checkImageMagick() {
  try {
    execSync('magick -version', { stdio: 'ignore' });
    return 'magick';
  } catch {
    try {
      execSync('convert -version', { stdio: 'ignore' });
      return 'convert';
    } catch {
      console.error('Error: ImageMagick is not installed.');
      console.error('Please install ImageMagick:');
      console.error('  macOS:   brew install imagemagick');
      console.error('  Ubuntu:  sudo apt-get install imagemagick');
      console.error('  Windows: choco install imagemagick');
      process.exit(1);
    }
  }
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function generateWindowsIco(magickCmd, masterIcon) {
  console.log('Generating Windows ICO...');
  const outputPath = path.join(buildDir, 'icon.ico');
  const sizes = '256,128,96,64,48,32,16';
  
  try {
    execSync(
      `${magickCmd} "${masterIcon}" -define icon:auto-resize=${sizes} "${outputPath}"`,
      { stdio: 'inherit' }
    );
    console.log(`✓ Created ${outputPath}`);
  } catch (error) {
    console.error('✗ Failed to create icon.ico');
  }

  console.log('Generating file icon...');
  const fileIconPath = path.join(buildDir, 'file-icon.ico');
  try {
    execSync(
      `${magickCmd} "${masterIcon}" -define icon:auto-resize=${sizes} "${fileIconPath}"`,
      { stdio: 'inherit' }
    );
    console.log(`✓ Created ${fileIconPath}`);
  } catch (error) {
    console.error('✗ Failed to create file-icon.ico');
  }
}

function generateMacOsIcns(magickCmd, masterIcon) {
  console.log('Generating macOS ICNS...');
  
  const iconsetDir = path.join(buildDir, 'icon.iconset');
  ensureDir(iconsetDir);

  const sizes = [
    { size: 16, name: 'icon_16x16.png' },
    { size: 32, name: 'icon_16x16@2x.png' },
    { size: 32, name: 'icon_32x32.png' },
    { size: 64, name: 'icon_32x32@2x.png' },
    { size: 128, name: 'icon_128x128.png' },
    { size: 256, name: 'icon_128x128@2x.png' },
    { size: 256, name: 'icon_256x256.png' },
    { size: 512, name: 'icon_256x256@2x.png' },
    { size: 512, name: 'icon_512x512.png' },
    { size: 1024, name: 'icon_512x512@2x.png' },
  ];

  for (const { size, name } of sizes) {
    const outputPath = path.join(iconsetDir, name);
    try {
      execSync(
        `${magickCmd} "${masterIcon}" -resize ${size}x${size} "${outputPath}"`,
        { stdio: 'ignore' }
      );
    } catch (error) {
      console.error(`✗ Failed to create ${name}`);
    }
  }

  if (process.platform === 'darwin') {
    try {
      const icnsPath = path.join(buildDir, 'icon.icns');
      execSync(`iconutil -c icns "${iconsetDir}" -o "${icnsPath}"`, { stdio: 'inherit' });
      console.log(`✓ Created ${icnsPath}`);
      
      execSync(`rm -rf "${iconsetDir}"`, { stdio: 'ignore' });
    } catch (error) {
      console.error('✗ Failed to create icon.icns (iconutil not available or failed)');
      console.log('  Keeping iconset directory for manual conversion');
    }
  } else {
    console.log('⚠ iconutil not available on this platform');
    console.log(`  Created iconset at ${iconsetDir}`);
    console.log('  Run "iconutil -c icns icon.iconset" on macOS to create icon.icns');
  }
}

function generateLinuxIcons(magickCmd, masterIcon) {
  console.log('Generating Linux icons...');
  
  ensureDir(iconsDir);

  const pngPath = path.join(buildDir, 'icon.png');
  try {
    execSync(`${magickCmd} "${masterIcon}" -resize 512x512 "${pngPath}"`, { stdio: 'ignore' });
    console.log(`✓ Created ${pngPath}`);
  } catch (error) {
    console.error('✗ Failed to create icon.png');
  }

  const sizes = [16, 32, 48, 64, 128, 256, 512];
  for (const size of sizes) {
    const outputPath = path.join(iconsDir, `${size}x${size}.png`);
    try {
      execSync(
        `${magickCmd} "${masterIcon}" -resize ${size}x${size} "${outputPath}"`,
        { stdio: 'ignore' }
      );
    } catch (error) {
      console.error(`✗ Failed to create ${size}x${size}.png`);
    }
  }
  console.log(`✓ Created icons in ${iconsDir}`);
}

function generatePlaceholder(magickCmd) {
  console.log('Generating placeholder icon...');
  
  const placeholderPath = path.join(buildDir, 'placeholder.png');
  
  try {
    execSync(
      `${magickCmd} -size 512x512 xc:'#007ACC' ` +
      `-font Arial-Bold -pointsize 200 -fill white ` +
      `-gravity center -annotate +0+0 'NA' "${placeholderPath}"`,
      { stdio: 'ignore' }
    );
    console.log(`✓ Created ${placeholderPath}`);
    return placeholderPath;
  } catch (error) {
    console.error('✗ Failed to create placeholder');
    return null;
  }
}

function generateDmgBackground(magickCmd) {
  console.log('Generating DMG background...');
  
  const bgPath = path.join(buildDir, 'background.png');
  
  try {
    execSync(
      `${magickCmd} -size 540x380 xc:'#1e1e1e' ` +
      `-font Arial -pointsize 24 -fill white ` +
      `-gravity center -annotate +0-50 'Neural Aquarium' ` +
      `-font Arial -pointsize 14 -fill '#aaaaaa' ` +
      `-gravity center -annotate +0+20 'Drag the icon to Applications folder' ` +
      `"${bgPath}"`,
      { stdio: 'ignore' }
    );
    console.log(`✓ Created ${bgPath}`);
  } catch (error) {
    console.error('✗ Failed to create background.png');
  }
}

function main() {
  const args = process.argv.slice(2);
  
  console.log('Neural Aquarium Icon Generator\n');
  
  const magickCmd = checkImageMagick();
  console.log(`Using ImageMagick: ${magickCmd}\n`);

  let masterIcon = args[0];
  
  if (!masterIcon) {
    console.log('No master icon provided, generating placeholder...\n');
    masterIcon = generatePlaceholder(magickCmd);
    if (!masterIcon) {
      console.error('Failed to generate placeholder icon');
      process.exit(1);
    }
    console.log('');
  }

  if (!fs.existsSync(masterIcon)) {
    console.error(`Error: Master icon not found: ${masterIcon}`);
    console.error('Usage: node generate-icons.js [path/to/master-icon.png]');
    process.exit(1);
  }

  console.log(`Master icon: ${masterIcon}\n`);

  generateWindowsIco(magickCmd, masterIcon);
  console.log('');
  
  generateMacOsIcns(magickCmd, masterIcon);
  console.log('');
  
  generateLinuxIcons(magickCmd, masterIcon);
  console.log('');
  
  generateDmgBackground(magickCmd);
  console.log('');

  console.log('✓ Icon generation complete!');
  console.log('\nGenerated files:');
  console.log('  icon.ico          - Windows application icon');
  console.log('  file-icon.ico     - Windows file association icon');
  console.log('  icon.icns         - macOS application icon');
  console.log('  icon.png          - Linux application icon');
  console.log('  icons/*.png       - Linux icon sizes');
  console.log('  background.png    - DMG background');
  
  if (masterIcon.includes('placeholder')) {
    console.log('\n⚠ WARNING: Using placeholder icons!');
    console.log('For production releases, replace with professional icons:');
    console.log('  node generate-icons.js path/to/professional-icon.png');
  }
}

main();
