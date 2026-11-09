# Security Policy

## Supported Versions

AnnoTuneAgent is under active development. Security fixes are applied to the
latest released version. Older releases are not maintained.

| Version | Supported |
| ------- | --------- |
| Latest release | :white_check_mark: |
| Older releases | :x: |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/leelieber2025/AnnoTuneAgent/security/advisories/new)
(repository → **Security** tab → **Report a vulnerability**).

Until the repository is public, email `leelieber@gmail.com`.

Include a description of the issue, steps to reproduce, and the affected
version.

## Scope

AnnoTuneAgent is a local single-cell analysis library. It does not send
telemetry. Annotation can call a local `llama-server` or an HTTP endpoint the
user supplies. Relevant concerns include unsafe deserialization of untrusted
input files (HDF5 / h5ad) and dependency vulnerabilities in packages it
relies on.
