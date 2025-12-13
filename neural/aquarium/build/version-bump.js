#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const args = process.argv.slice(2);
const versionType = args[0] || 'patch';

const validTypes = ['major', 'minor', 'patch', 'premajor', 'preminor', 'prepatch', 'prerelease'];
if (!validTypes.includes(versionType) && !versionType.match(/^\d+\.\d+\.\d+/)) {
  console.error(`Invalid version type: ${versionType}`);
  console.error(`Valid types: ${validTypes.join(', ')} or explicit version (e.g., 1.2.3)`);
  process.exit(1);
}

const projectRoot = path.resolve(__dirname, '..');

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n', 'utf8');
}

function getCurrentVersion() {
  const packageJson = readJson(path.join(projectRoot, 'package.json'));
  return packageJson.version;
}

function bumpVersion(currentVersion, type) {
  if (type.match(/^\d+\.\d+\.\d+/)) {
    return type;
  }

  const [major, minor, patch] = currentVersion.split('.').map(Number);

  switch (type) {
    case 'major':
      return `${major + 1}.0.0`;
    case 'minor':
      return `${major}.${minor + 1}.0`;
    case 'patch':
      return `${major}.${minor}.${patch + 1}`;
    case 'premajor':
      return `${major + 1}.0.0-alpha.0`;
    case 'preminor':
      return `${major}.${minor + 1}.0-alpha.0`;
    case 'prepatch':
      return `${major}.${minor}.${patch + 1}-alpha.0`;
    case 'prerelease':
      if (currentVersion.includes('-')) {
        const [base, pre] = currentVersion.split('-');
        const [label, num] = pre.split('.');
        return `${base}-${label}.${parseInt(num) + 1}`;
      }
      return `${major}.${minor}.${patch + 1}-alpha.0`;
    default:
      throw new Error(`Unknown version type: ${type}`);
  }
}

function updatePackageJson(newVersion) {
  const packageJsonPath = path.join(projectRoot, 'package.json');
  const packageJson = readJson(packageJsonPath);
  packageJson.version = newVersion;
  writeJson(packageJsonPath, packageJson);
  console.log(`✓ Updated package.json to ${newVersion}`);
}

function updateElectronPackageJson(newVersion) {
  const electronPackagePath = path.join(projectRoot, 'electron', 'package.json');
  if (fs.existsSync(electronPackagePath)) {
    const electronPackage = readJson(electronPackagePath);
    electronPackage.version = newVersion;
    writeJson(electronPackagePath, electronPackage);
    console.log(`✓ Updated electron/package.json to ${newVersion}`);
  }
}

function updateChangelog(newVersion) {
  const changelogPath = path.join(projectRoot, 'CHANGELOG.md');
  if (!fs.existsSync(changelogPath)) {
    console.log('⚠ CHANGELOG.md not found, skipping');
    return;
  }

  let changelog = fs.readFileSync(changelogPath, 'utf8');
  const date = new Date().toISOString().split('T')[0];
  const heading = `## [${newVersion}] - ${date}`;

  if (changelog.includes('## [Unreleased]')) {
    changelog = changelog.replace('## [Unreleased]', `## [Unreleased]\n\n${heading}`);
  } else {
    const lines = changelog.split('\n');
    const insertIndex = lines.findIndex(line => line.startsWith('## ['));
    if (insertIndex > 0) {
      lines.splice(insertIndex, 0, heading, '');
      changelog = lines.join('\n');
    } else {
      changelog = `${heading}\n\n${changelog}`;
    }
  }

  fs.writeFileSync(changelogPath, changelog, 'utf8');
  console.log(`✓ Updated CHANGELOG.md with ${newVersion}`);
}

function gitCommit(newVersion) {
  try {
    execSync('git add package.json electron/package.json CHANGELOG.md', { stdio: 'inherit' });
    execSync(`git commit -m "Bump version to ${newVersion}"`, { stdio: 'inherit' });
    console.log(`✓ Created git commit`);
  } catch (error) {
    console.log('⚠ Git commit failed, continuing...');
  }
}

function gitTag(newVersion, push = false) {
  try {
    const tagName = `aquarium-v${newVersion}`;
    execSync(`git tag -a ${tagName} -m "Release ${newVersion}"`, { stdio: 'inherit' });
    console.log(`✓ Created git tag ${tagName}`);

    if (push) {
      execSync(`git push origin ${tagName}`, { stdio: 'inherit' });
      console.log(`✓ Pushed tag ${tagName}`);
    }
  } catch (error) {
    console.log('⚠ Git tag failed, continuing...');
  }
}

function main() {
  console.log('Neural Aquarium Version Bump\n');

  const currentVersion = getCurrentVersion();
  console.log(`Current version: ${currentVersion}`);

  const newVersion = bumpVersion(currentVersion, versionType);
  console.log(`New version: ${newVersion}\n`);

  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });

  readline.question('Continue? (y/n) ', (answer) => {
    readline.close();

    if (answer.toLowerCase() !== 'y') {
      console.log('Aborted');
      process.exit(0);
    }

    console.log('\nUpdating files...\n');

    updatePackageJson(newVersion);
    updateElectronPackageJson(newVersion);
    updateChangelog(newVersion);

    if (args.includes('--commit')) {
      gitCommit(newVersion);
    }

    if (args.includes('--tag')) {
      const push = args.includes('--push');
      gitTag(newVersion, push);
    }

    console.log('\n✓ Version bump complete!');
    console.log(`\nNext steps:`);
    console.log(`  1. Review changes: git diff`);
    console.log(`  2. Commit: git commit -am "Release ${newVersion}"`);
    console.log(`  3. Tag: git tag -a aquarium-v${newVersion} -m "Release ${newVersion}"`);
    console.log(`  4. Push: git push origin aquarium-v${newVersion}`);
  });
}

main();
