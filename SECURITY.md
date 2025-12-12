# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take the security of Neural DSL seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:
- **Email**: Lemniscate_zero@proton.me
- **Subject**: [SECURITY] Brief description of the issue

### What to Include

Please include the following information in your report:

1. **Type of issue** (e.g., code injection, information disclosure, etc.)
2. **Full paths of source file(s)** related to the manifestation of the issue
3. **Location of the affected source code** (tag/branch/commit or direct URL)
4. **Any special configuration** required to reproduce the issue
5. **Step-by-step instructions** to reproduce the issue
6. **Proof-of-concept or exploit code** (if possible)
7. **Impact of the issue**, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 5 business days with our evaluation and expected timeline
- **Resolution**: Varies based on severity, but we aim for:
  - Critical vulnerabilities: 7 days
  - High severity: 30 days
  - Medium/Low severity: 90 days

### Disclosure Policy

- We follow a **coordinated disclosure** policy
- Please give us reasonable time to address the issue before any public disclosure
- We will credit you in our security advisory (unless you prefer to remain anonymous)
- Once a fix is available, we will:
  1. Release a patch
  2. Publish a security advisory
  3. Update the CHANGELOG.md with security notes

### Security Measures

Neural DSL implements several security measures:

#### Authentication
- HTTP Basic Auth for API endpoints and WebSocket connections
- Credentials configurable via `config.yaml`
- Never commit credentials to the repository

#### CI/CD Security
- **Bandit**: Static analysis for common security issues in Python code
- **Safety**: Checks dependencies against known security vulnerabilities
- **pip-audit**: Audits Python packages for known vulnerabilities
- **git-secrets**: Prevents committing secrets to the repository
- **TruffleHog**: Scans for secrets in git history

#### Best Practices
- All user inputs are validated before processing
- No arbitrary code execution from DSL definitions
- File operations are restricted to designated directories
- Dependencies are regularly updated and audited

### Security Features

#### Hacky Mode
The `--hacky` flag enables security analysis features:
- **Gradient Leakage Detection**: Identifies potential information leakage
- **Adversarial Input Testing**: Tests robustness against malicious inputs
- **Usage**: `neural debug my_model.neural --hacky`

**Warning**: Use hacky mode only in controlled environments for security research purposes.

### Scope

The following are **in scope** for vulnerability reports:
- Code injection vulnerabilities
- Authentication/authorization bypass
- Information disclosure
- Denial of service vulnerabilities
- Dependency vulnerabilities with exploits
- Secret exposure in code or logs

The following are **out of scope**:
- Social engineering attacks
- Physical attacks
- Vulnerabilities in third-party dependencies without demonstrable exploit
- Issues in example code not meant for production
- Issues requiring physical access to the system

### Bug Bounty

We do not currently offer a paid bug bounty program. However, we will:
- Publicly acknowledge your contribution (with permission)
- Add your name to our security contributors list
- Provide a detailed security advisory crediting your discovery

## Security Updates

Security updates are announced through:
- GitHub Security Advisories
- Release notes in CHANGELOG.md
- Email notifications (for critical vulnerabilities)

## Additional Resources

- [Flask Security Best Practices](https://flask.palletsprojects.com/en/latest/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

## Contact

For general security questions or concerns:
- **Email**: Lemniscate_zero@proton.me
- **GitHub**: [@Lemniscate-SHA-256](https://github.com/Lemniscate-SHA-256)

---

**Last Updated**: 2024
