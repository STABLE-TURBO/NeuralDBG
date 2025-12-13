# Code Signing Setup Guide

This document provides instructions for setting up code signing for Neural Aquarium across different platforms.

## Windows Code Signing

### Requirements
- A valid code signing certificate (EV or standard)
- Certificate file (.pfx or .p12)
- Certificate password

### Setup

1. **Obtain a Code Signing Certificate**
   - Purchase from a trusted CA (DigiCert, Sectigo, GlobalSign, etc.)
   - For EV certificates, you'll receive a USB token
   - For standard certificates, you'll receive a .pfx/.p12 file

2. **Environment Variables**
   ```bash
   # Certificate file path
   export WIN_CSC_LINK=/path/to/certificate.pfx
   
   # Certificate password
   export WIN_CSC_KEY_PASSWORD=your-password
   ```

3. **GitHub Actions Secrets**
   - `WIN_CSC_LINK`: Base64-encoded certificate file
   - `WIN_CSC_KEY_PASSWORD`: Certificate password
   
   To encode certificate:
   ```bash
   cat certificate.pfx | base64 > certificate.txt
   ```

4. **Alternative: Azure Key Vault**
   For enhanced security, use Azure Key Vault:
   ```bash
   export AZURE_KEY_VAULT_URI=https://your-vault.vault.azure.net/
   export AZURE_KEY_VAULT_CLIENT_ID=your-client-id
   export AZURE_KEY_VAULT_CLIENT_SECRET=your-client-secret
   export AZURE_KEY_VAULT_TENANT_ID=your-tenant-id
   export AZURE_KEY_VAULT_CERTIFICATE=your-cert-name
   ```

### Verification
```bash
# Verify signature
signtool verify /pa /v "Neural Aquarium-0.3.0-win-x64.exe"
```

## macOS Code Signing

### Requirements
- Apple Developer account ($99/year)
- Developer ID Application certificate
- App-specific password for notarization
- Xcode Command Line Tools

### Setup

1. **Create Certificates**
   - Log in to [Apple Developer](https://developer.apple.com)
   - Navigate to Certificates, Identifiers & Profiles
   - Create "Developer ID Application" certificate
   - Download and install in Keychain Access

2. **Find Certificate Identity**
   ```bash
   security find-identity -v -p codesigning
   ```
   Look for "Developer ID Application: Your Name (TEAM_ID)"

3. **App-Specific Password**
   - Go to [appleid.apple.com](https://appleid.apple.com)
   - Sign in > Security > App-Specific Passwords
   - Generate new password
   - Save securely

4. **Environment Variables**
   ```bash
   # Apple ID for notarization
   export APPLE_ID=your@email.com
   
   # App-specific password
   export APPLE_ID_PASSWORD=xxxx-xxxx-xxxx-xxxx
   
   # Team ID
   export APPLE_TEAM_ID=YOUR_TEAM_ID
   
   # Certificate identity (optional, auto-detected if in keychain)
   export APPLE_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"
   
   # Provisioning profile (optional)
   export MAC_PROVISIONING_PROFILE=/path/to/profile.provisionprofile
   ```

5. **GitHub Actions Secrets**
   - `APPLE_ID`: Your Apple ID email
   - `APPLE_ID_PASSWORD`: App-specific password
   - `APPLE_TEAM_ID`: Your team ID
   - `APPLE_CERTIFICATE`: Base64-encoded .p12 certificate
   - `APPLE_CERTIFICATE_PASSWORD`: Certificate password
   
   Export certificate from Keychain:
   ```bash
   # Export from Keychain as .p12
   # Then encode:
   cat certificate.p12 | base64 > certificate.txt
   ```

### Notarization Process
The build process automatically:
1. Signs the application with your Developer ID
2. Creates a DMG
3. Submits to Apple for notarization
4. Staples the notarization ticket to the DMG

### Verification
```bash
# Check code signature
codesign --verify --deep --strict --verbose=2 "Neural Aquarium.app"

# Check notarization
spctl --assess --verbose=4 --type execute "Neural Aquarium.app"

# Check stapled ticket
stapler validate "Neural Aquarium.dmg"
```

## Linux Code Signing

Linux uses GPG signing for packages.

### Setup

1. **Generate GPG Key**
   ```bash
   gpg --full-generate-key
   # Choose RSA, 4096 bits, no expiration
   # Enter name and email matching package maintainer
   ```

2. **Export Public Key**
   ```bash
   gpg --armor --export your@email.com > public.key
   ```

3. **Environment Variables**
   ```bash
   # GPG key ID
   export GPG_KEY_ID=YOUR_KEY_ID
   
   # GPG passphrase
   export GPG_PASSPHRASE=your-passphrase
   ```

4. **Sign Packages**
   Electron Builder automatically signs .deb and .rpm packages if GPG is configured.

5. **AppImage Signing**
   ```bash
   # Sign AppImage
   gpg --detach-sign "Neural Aquarium-0.3.0-linux-x64.AppImage"
   
   # Verify
   gpg --verify "Neural Aquarium-0.3.0-linux-x64.AppImage.sig"
   ```

## CI/CD Integration

### GitHub Actions Secrets Required

**Windows:**
- `WIN_CSC_LINK`: Base64-encoded certificate
- `WIN_CSC_KEY_PASSWORD`: Certificate password

**macOS:**
- `APPLE_ID`: Apple ID email
- `APPLE_ID_PASSWORD`: App-specific password
- `APPLE_TEAM_ID`: Team ID
- `APPLE_CERTIFICATE`: Base64-encoded P12 certificate
- `APPLE_CERTIFICATE_PASSWORD`: Certificate password

**Linux:**
- `GPG_PRIVATE_KEY`: GPG private key (armor format)
- `GPG_PASSPHRASE`: GPG passphrase

### Security Best Practices

1. **Never commit certificates or keys to repository**
2. **Use environment variables or CI/CD secrets**
3. **Rotate certificates before expiration**
4. **Use hardware tokens (EV certificates) when possible**
5. **Enable 2FA on all accounts**
6. **Use separate certificates for production and testing**
7. **Monitor certificate expiration dates**
8. **Keep backup copies of certificates in secure location**

## Testing Without Certificates

For development and testing without certificates:

```bash
# Skip code signing
export CSC_IDENTITY_AUTO_DISCOVERY=false

# Build without signing
npm run electron:build
```

## Troubleshooting

### Windows
- **Error: "Certificate not found"**: Check WIN_CSC_LINK path and password
- **Timestamp server errors**: Network issues, retry or use different timestamp server
- **"Not signed" warning**: Verify signtool.exe is in PATH

### macOS
- **"No identity found"**: Import certificate to Keychain, verify with security command
- **Notarization timeout**: Large apps take longer, can take up to 1 hour
- **Gatekeeper rejection**: Check entitlements and hardened runtime settings
- **"Unable to notarize"**: Verify Apple ID and app-specific password

### Linux
- **GPG signing fails**: Check GPG_KEY_ID and passphrase
- **Package verification fails**: Ensure public key is distributed with package

## Cost Estimates

- **Windows EV Certificate**: $300-500/year
- **Windows Standard Certificate**: $100-300/year
- **Apple Developer Program**: $99/year
- **Linux GPG**: Free

## Resources

- [Windows Code Signing Best Practices](https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)
- [Apple Notarization Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Electron Builder Code Signing](https://www.electron.build/code-signing)
- [GPG Documentation](https://gnupg.org/documentation/)
