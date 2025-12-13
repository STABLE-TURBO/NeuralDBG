#!/usr/bin/env python3
"""
Create an example npm plugin package structure
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
NPM_PLUGIN_DIR = BASE_DIR / "src" / "plugins" / "examples" / "npm_plugin_example"

NPM_PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

files = {}

# ============================================================================
# NPM PLUGIN EXAMPLE - Complete package structure
# ============================================================================

files[NPM_PLUGIN_DIR / "package.json"] = '''{
  "name": "@neural-aquarium/example-panel-plugin",
  "version": "1.0.0",
  "description": "Example panel plugin for Neural Aquarium demonstrating npm integration",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "author": "Neural DSL Team",
  "license": "MIT",
  "keywords": [
    "neural-aquarium",
    "plugin",
    "panel",
    "example"
  ],
  "repository": {
    "type": "git",
    "url": "https://github.com/neural-dsl/plugins"
  },
  "homepage": "https://github.com/neural-dsl/plugins/tree/main/npm-panel-example",
  "neuralAquariumPlugin": {
    "displayName": "Example Panel Plugin",
    "capabilities": ["panel"],
    "minAquariumVersion": "0.3.0",
    "icon": "📋"
  },
  "scripts": {
    "build": "tsc",
    "prepublishOnly": "npm run build"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^4.9.0"
  },
  "peerDependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  }
}
'''

files[NPM_PLUGIN_DIR / "tsconfig.json"] = '''{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM"],
    "jsx": "react",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
'''

files[NPM_PLUGIN_DIR / "src" / "index.tsx"] = '''import React from 'react';
import { Plugin, PanelPlugin } from '@neural-aquarium/plugin-api';

interface ExamplePanelProps {
  data?: any;
}

const ExamplePanel: React.FC<ExamplePanelProps> = ({ data }) => {
  return (
    <div style={{
      padding: '20px',
      backgroundColor: '#f5f5f5',
      borderRadius: '8px',
      height: '100%'
    }}>
      <h2 style={{ margin: '0 0 16px 0', color: '#333' }}>
        📋 Example Panel
      </h2>
      <p style={{ color: '#666', lineHeight: 1.6 }}>
        This is an example panel plugin installed from npm.
        It demonstrates how to create custom UI panels for Neural Aquarium.
      </p>
      <div style={{
        marginTop: '20px',
        padding: '12px',
        backgroundColor: 'white',
        borderRadius: '4px',
        border: '1px solid #ddd'
      }}>
        <strong>Plugin Info:</strong>
        <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
          <li>Installed from npm</li>
          <li>Written in TypeScript + React</li>
          <li>Fully type-safe</li>
          <li>Easy to customize</li>
        </ul>
      </div>
      {data && (
        <div style={{ marginTop: '16px' }}>
          <strong>Received Data:</strong>
          <pre style={{
            marginTop: '8px',
            padding: '12px',
            backgroundColor: '#1e1e1e',
            color: '#d4d4d4',
            borderRadius: '4px',
            overflow: 'auto',
            fontSize: '12px'
          }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

export class ExamplePanelPlugin implements PanelPlugin {
  private enabled = false;

  async initialize(): Promise<void> {
    console.log('Example Panel Plugin initialized');
  }

  async activate(): Promise<void> {
    this.enabled = true;
    console.log('Example Panel Plugin activated');
  }

  async deactivate(): Promise<void> {
    this.enabled = false;
    console.log('Example Panel Plugin deactivated');
  }

  getPanelComponent(): React.ComponentType<any> {
    return ExamplePanel;
  }

  getPanelConfig(): any {
    return {
      title: 'Example Panel',
      position: 'right',
      width: 400,
      resizable: true,
      closeable: true,
      icon: '📋'
    };
  }
}

export default ExamplePanelPlugin;
'''

files[NPM_PLUGIN_DIR / "src" / "types.d.ts"] = '''declare module '@neural-aquarium/plugin-api' {
  export interface Plugin {
    initialize(): Promise<void>;
    activate(): Promise<void>;
    deactivate(): Promise<void>;
  }

  export interface PanelPlugin extends Plugin {
    getPanelComponent(): React.ComponentType<any>;
    getPanelConfig(): any;
  }
}
'''

files[NPM_PLUGIN_DIR / "README.md"] = '''# Example Panel Plugin for Neural Aquarium

An example plugin demonstrating how to create custom UI panels for Neural Aquarium using npm packages.

## Installation

```bash
npm install @neural-aquarium/example-panel-plugin
```

Or install directly in Neural Aquarium:
1. Open Plugin Marketplace
2. Search for "Example Panel Plugin"
3. Click Install

## Features

- Custom React-based UI panel
- TypeScript for type safety
- Configurable layout
- Data visualization support
- Easy to extend

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Publish
npm publish
```

## Usage

The plugin is automatically loaded by Neural Aquarium when installed. Enable it from the Plugin Marketplace to see the example panel.

## Creating Your Own Plugin

Use this as a template:

1. Copy this directory structure
2. Update package.json with your plugin info
3. Modify src/index.tsx with your panel logic
4. Build and publish to npm
5. Install in Neural Aquarium

## API Reference

### PanelPlugin Interface

```typescript
interface PanelPlugin {
  initialize(): Promise<void>;
  activate(): Promise<void>;
  deactivate(): Promise<void>;
  getPanelComponent(): React.ComponentType<any>;
  getPanelConfig(): PanelConfig;
}
```

### PanelConfig

```typescript
interface PanelConfig {
  title: string;
  position: 'left' | 'right' | 'top' | 'bottom';
  width?: number;
  height?: number;
  resizable?: boolean;
  closeable?: boolean;
  icon?: string;
}
```

## License

MIT
'''

files[NPM_PLUGIN_DIR / ".npmignore"] = '''src/
tsconfig.json
*.ts
!*.d.ts
node_modules/
.vscode/
.DS_Store
'''

files[NPM_PLUGIN_DIR / "PUBLISHING.md"] = '''# Publishing to npm

## Prerequisites

1. npm account: https://www.npmjs.com/signup
2. Verified email
3. Configured npm authentication: `npm login`

## Steps

### 1. Prepare Package

```bash
# Ensure package.json is correct
npm version patch  # or minor, major

# Build the package
npm run build

# Test locally
npm pack
# This creates a .tgz file you can test with:
# npm install ./neural-aquarium-example-panel-plugin-1.0.0.tgz
```

### 2. Publish

```bash
# Public package (recommended for open source)
npm publish --access public

# Private package (requires paid npm account)
npm publish
```

### 3. Verify

```bash
# Check package page
open https://www.npmjs.com/package/@neural-aquarium/example-panel-plugin

# Test installation
npm install @neural-aquarium/example-panel-plugin
```

## Versioning

Follow semantic versioning:
- **Patch** (1.0.x): Bug fixes
- **Minor** (1.x.0): New features, backward compatible
- **Major** (x.0.0): Breaking changes

```bash
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0
```

## Best Practices

1. **Test before publishing**: Always run tests and build locally
2. **Use .npmignore**: Don't include source files in published package
3. **Include README**: Good documentation increases adoption
4. **Tag releases**: Create git tags for versions
5. **Changelog**: Maintain CHANGELOG.md
6. **License**: Include LICENSE file
7. **Keywords**: Use relevant keywords in package.json
8. **Scope**: Use organization scope (@neural-aquarium/)

## Troubleshooting

### Already Published Version

```bash
# Increment version and try again
npm version patch
npm publish
```

### Permission Denied

```bash
# Login again
npm login

# Check who you're logged in as
npm whoami
```

### Package Name Taken

- Choose a different name
- Use a scoped package: @yourname/package-name

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Publish to npm

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'
      - run: npm ci
      - run: npm run build
      - run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

## Support

- npm documentation: https://docs.npmjs.com/
- Neural Aquarium plugins: https://github.com/neural-dsl/plugins
'''

# Write all files
for filepath, content in files.items():
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\n✅ NPM plugin example created successfully!")
print(f"   Location: {NPM_PLUGIN_DIR}")
print("\n📦 This demonstrates:")
print("   • Complete npm package structure")
print("   • TypeScript + React integration")
print("   • Publishing workflow")
print("   • Best practices")
