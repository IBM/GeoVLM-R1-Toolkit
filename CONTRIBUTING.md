## Contributing In General
Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a [pull request](https://github.com/IBM/GeoVLM-R1-Toolkit/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue tracker](https://github.com/IBM/GeoVLM-R1-Toolkit/issues).
Before embarking on a more ambitious contribution, please quickly [get in touch](#communication) with us.

**Note: We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in backlog for a long time, or
cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please [raise an issue](https://github.com/IBM/GeoVLM-R1-Toolkit/issues)
before sending a pull request so the feature can be discussed. This is to avoid
you wasting your valuable time working on a feature that the project developers
are not interested in accepting into the code base.

### Fixing bugs

If you would like to fix a bug, please [raise an issue](https://github.com/IBM/GeoVLM-R1-Toolkit/issues) before sending a
pull request so it can be tracked.

### Merge approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. A change requires an LGTM from at least one
maintainer of each component affected.

For a list of the maintainers, see the [MAINTAINERS.md](MAINTAINERS.md) page.

## Legal

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License (CC BY-NC 4.0) and is intended for **non-commercial research
purposes only**. By contributing, you agree that your contributions are licensed
under the same terms.

Each new source file must include a copyright and license header. Using the SPDX
format is the simplest approach, e.g.

```
#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
```

Files derived from third-party code must keep their original copyright notice
(see [NOTICE](NOTICE)).

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) - that the Linux® Kernel [community](https://elinux.org/Developer_Certificate_Of_Origin)
uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the
submitter accepts the DCO:

```
Signed-off-by: John Doe <john.doe@example.com>
```

You can include this automatically when you commit a change to your
local git repository using the following command:

```
git commit -s
```

## Communication
Please open a [GitHub issue](https://github.com/IBM/GeoVLM-R1-Toolkit/issues) for questions,
bug reports and feature requests, or contact the maintainer at
[mustansar.fiaz@ibm.com](mailto:mustansar.fiaz@ibm.com).

## Setup
Follow the installation steps in the [README](README.md#1-installation). In short:

```bash
git clone https://github.com/IBM/GeoVLM-R1-Toolkit.git
cd GeoVLM-R1-Toolkit
conda create -n geovlm-r1 python=3.11 -y && conda activate geovlm-r1
cd train/GeoVLM_R1 && bash setup.sh
```

## Testing
Training and inference need GPUs. Before submitting a pull request, please at least:

```bash
# all Python files compile
python -m compileall -q GeoVLM_Eval GeoVLM_inference train
# all shell scripts parse
for f in $(find . -name '*.sh' -not -path './.git/*'); do bash -n "$f"; done
```

If you change an evaluation script, run it on a small prediction file
(`bash GeoVLM_Eval/eval.sh`, or a single `rs_<task>/eval.py --output_path <dir>`)
and include the printed metrics in the pull request.

## Coding style guidelines
- Python: follow [PEP 8](https://peps.python.org/pep-0008/). Formatting with `black`/`ruff` is
  welcome (the GRPO package provides `make style` / `make quality` in `train/GeoVLM_R1`).
- Do not commit hard-coded absolute paths, credentials, datasets, model weights or result files.
  Use relative defaults and command-line arguments.
- Keep the copyright/SPDX header at the top of every source file.
