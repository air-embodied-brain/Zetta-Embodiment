# Zetta Documentation

Zetta's documentation currently lives as Markdown files in this directory.

| Document | Topic |
|---|---|
| [behavior.md](behavior.md) | BEHAVIOR / OmniGibson Runtime backend |
| [cosmos-lite.md](cosmos-lite.md) | Cosmos-Lite remote policy backend (中文) |
| [geniesim.md](geniesim.md) | Genie Sim / Isaac Sim Runtime |
| [geniesim-vla.md](geniesim-vla.md) | Genie Sim VLA and Campaign |
| [maniskill.md](maniskill.md) | Native ManiSkill support |
| [libero-recovery-tools.md](libero-recovery-tools.md) | LIBERO recovery tool additions |
| [img-encode-report/nvjpeg-optimization-report.md](img-encode-report/nvjpeg-optimization-report.md) | nvJPEG image codec performance report (中文) |

The VLA runtime setup guide for LIBERO-Pro and RoboCasa lives next to its
install script at
[`scripts/deployment/VLA_ENV_SETUP.md`](../scripts/deployment/VLA_ENV_SETUP.md).

## Planned Sphinx site

A Sphinx site with parallel English and Simplified Chinese sources
(`source-en/` and `source-zh/`) is planned but not yet available. The
`Makefile`, `autobuild.sh`, and `requirements.txt` in this directory are
scaffolding for it and do not build anything until those source trees are
added.
