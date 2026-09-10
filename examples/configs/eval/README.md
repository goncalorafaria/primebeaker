# JTC evaluation lifecycle YAMLs

PrimeBeaker owns these lifecycle descriptions and calls the released JTC
Python workers inside the configured image. Standard configs omit `workflow`;
search rollouts use `workflow: search-agent`; post-hoc audits use
`workflow: rubric-judge-audit`.

Every stable workflow in `jtc_data_commons.workflows.registry` can also be the
top-level `workflow` value. Put its Python `main()` keyword arguments under
`arguments`, and describe the LiteRegistry capacity plus readiness names under
`services` and `required_services`. This covers all verifier variants,
TMAX/Podman and binary-submit evaluators, inference, rubric generation, and
future workflows added to the JTC registry—not only search-agent evaluation.

Representative generic configs live under `verifier/` and `workflow/`.
PrimeBeaker replaces workflow endpoint arguments with its managed registry and
gateway, so these YAMLs never launch their own gateway.

Iterative rejection sampling uses `workflow: rejection-sampling`. Its
`rejection` block contains only the JTC judge/filter/compile behavior, while
the sibling `services`, placement, readiness, secrets, image, and cleanup
fields are PrimeBeaker-owned. Each completed round has a receipt and can be
resumed without fixed ports or lifecycle shell scripts. See
`rejection_sampling/verifier_rejection_sampling_smoke.yaml`.

List the names accepted by the installed JTC release:

```bash
primebeaker evaluation workflows
```

```bash
primebeaker evaluation preview --config examples/configs/eval/python_only_smoke.yaml
primebeaker evaluation submit --config examples/configs/eval/python_only_smoke.yaml
```

Override the image or reuse a registry without editing a YAML:

```bash
primebeaker evaluation preview \
  --config examples/configs/eval/python_only_smoke.yaml \
  --image beaker://IMMUTABLE_JTC_EVAL_IMAGE \
  --registry redis://registry-host:6379
```

Template paths may be relative to this YAML tree, relative to a checkout, an
absolute mounted path, or a template bundled in the JTC wheel. PrimeBeaker
validates the JSON and snapshots it under the run directory before submission,
so a newly added template works without rebuilding the evaluation image.
