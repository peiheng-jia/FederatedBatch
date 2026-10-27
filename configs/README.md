# Experiment configurations

Use these five top-level YAML files for the documented workflows:

- `figure1.yaml`: Figure 1 main experiment.
- `figure2.yaml`: Figure 2 canonical training configuration.
- `figure3_4.yaml`: shared configuration for Figures 3 and 4.
- `figure3_4_fedscgen.yaml`: method-specific parameters for the FedscGen baseline used in Figures 3 and 4.
- `figure5.yaml`: Figure 5 privacy attacks and privacy accounting.

The Figure 3/4 driver uses both the shared configuration and the FedscGen
configuration. Figure 5 requires the Figure 2 results and a separately prepared
SecretFlow-SPU environment for the secure-computation step.

## Legacy and alternative configurations

Files under `legacy/` are not the default entry points:

- `legacy/figure2_freshqueue_baseline.yaml`: retained only for the explicit
  FreshQueue comparator used by the Figure 5 workflow.
- `legacy/figure5_final_model_privacy.yaml`: an alternative configuration for
  privacy analysis using the bundled final Figure 2 results; it is not used by
  the documented canonical Figure 5 commands.

For executable commands, follow
[reproduction guide](../docs/reproduction.md).
