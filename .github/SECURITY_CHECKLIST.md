# Security Review Checklist

Use this checklist when reviewing code for security issues or setting up security scanning.

## Pre-Commit Checklist

- [ ] Run `pre-commit run --all-files` before committing
- [ ] No secrets committed (API keys, passwords, tokens)
- [ ] Bandit scan shows no critical/high severity issues
- [ ] Safety check passes or known vulnerabilities documented
- [ ] No hardcoded credentials in code or config files
- [ ] Environment variables used for sensitive configuration
- [ ] `.env.example` updated if new env vars added

## Code Review Checklist

### Input Validation
- [ ] All user inputs are validated and sanitized
- [ ] No SQL injection vulnerabilities (use parameterized queries)
- [ ] No command injection (avoid shell=True with user input)
- [ ] File paths validated to prevent directory traversal
- [ ] File upload restrictions in place (size, type, content)

### Authentication & Authorization
- [ ] Authentication required for sensitive endpoints
- [ ] Authorization checks properly implemented
- [ ] Credentials never logged or exposed in responses
- [ ] Session management properly configured
- [ ] Password requirements enforced (if applicable)

### Data Security
- [ ] Sensitive data encrypted at rest and in transit
- [ ] No sensitive data in logs or error messages
- [ ] Proper error handling without information disclosure
- [ ] Database credentials stored securely
- [ ] API keys and secrets use environment variables

### Code Quality
- [ ] No use of `eval()`, `exec()`, or `__import__()` with user input
- [ ] No use of `pickle` with untrusted data
- [ ] No hardcoded secrets or credentials
- [ ] Dependencies up to date and audited
- [ ] No debug code or commented-out credentials

### Network Security
- [ ] HTTPS enforced for production
- [ ] CORS properly configured
- [ ] Rate limiting implemented for APIs
- [ ] Timeouts configured for external requests
- [ ] Certificate validation enabled

## CI/CD Checklist

### Workflow Configuration
- [ ] Security scanning workflows configured
- [ ] Scheduled audits enabled (weekly recommended)
- [ ] Workflow artifacts uploaded for review
- [ ] Critical vulnerabilities block merge (optional)

### Tools Enabled
- [ ] Bandit (Python code security)
- [ ] Safety (dependency vulnerabilities)
- [ ] pip-audit (package vulnerabilities)
- [ ] git-secrets (secret scanning)
- [ ] TruffleHog (git history scanning)
- [ ] detect-secrets (pre-commit hook)

### Monitoring
- [ ] Dependabot or similar for dependency updates
- [ ] GitHub Security Advisories enabled
- [ ] CodeQL or similar SAST tool configured
- [ ] Security alerts notifications configured

## Deployment Checklist

### Pre-Deployment
- [ ] All security scans passing or issues documented
- [ ] Dependencies audited and updated
- [ ] Security configuration reviewed
- [ ] Environment variables properly set
- [ ] Debug mode disabled in production

### Post-Deployment
- [ ] Security monitoring enabled
- [ ] Log analysis configured
- [ ] Intrusion detection configured (if applicable)
- [ ] Backup and recovery tested
- [ ] Incident response plan documented

## Dependency Management

### Before Adding Dependencies
- [ ] Check package reputation and maintainer
- [ ] Review package for known vulnerabilities
- [ ] Evaluate package necessity (avoid over-dependencies)
- [ ] Check package license compatibility
- [ ] Pin version or use version range

### Regular Maintenance
- [ ] Review and update dependencies monthly
- [ ] Remove unused dependencies
- [ ] Audit dependencies with pip-audit
- [ ] Review Dependabot alerts promptly
- [ ] Test after dependency updates

## Documentation

- [ ] Security policy (SECURITY.md) up to date
- [ ] Security setup guide accurate
- [ ] Vulnerability reporting process documented
- [ ] Security best practices documented
- [ ] Known security limitations documented

## Incident Response

- [ ] Vulnerability reporting email monitored
- [ ] Incident response plan exists
- [ ] Security contact information current
- [ ] Coordinated disclosure process defined
- [ ] Patch release process documented

## Compliance

- [ ] License compliance verified
- [ ] Data privacy requirements met (GDPR, etc.)
- [ ] Export control requirements checked (if applicable)
- [ ] Security standards followed (OWASP, etc.)
- [ ] Audit trail requirements met

## Periodic Reviews (Quarterly)

- [ ] Review and update security checklist
- [ ] Review security tool configurations
- [ ] Review and update security documentation
- [ ] Review access controls and permissions
- [ ] Review and test incident response plan
- [ ] Review dependency security
- [ ] Review and update security training

## Notes

- Mark N/A for items not applicable to your project
- Document exceptions and accepted risks
- Keep this checklist updated as threats evolve
- Use issue templates to track security tasks

---

**Last Updated**: 2024
**Next Review**: [Schedule quarterly review]
