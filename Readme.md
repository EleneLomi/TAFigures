Repository containing learning dynamics of predictive coding and backpropagation with and without rescaling.

Five folders contain code, figures and csv files for figures 2, 3, 4, 5 and the nonlinear autoencoders. Within each folder, 'output_data' usually contains the data used for the figures (one hidden layer network) and the folder titled 'Deep' contains experiments with deeper networks (eight hidden layers).

## Single training step panels

The "Single training step" panels are in a `SingleStep` folder inside each figure's folder, with their data in `SingleStep/output_data`.

| Panel | Script | Shows |
| --- | --- | --- |
| Figure 2 A, C | `Figure2/SingleStep/panels_ABCD_single_step.py` | Alignment vs depth (A: Kaiming, C: Norm-preserving) |
| Figure 2 B, D | `Figure2/SingleStep/panels_ABCD_single_step.py` | Alignment vs width (B: Kaiming, D: Norm-preserving) |
| Figure 3 A | `Figure3/SingleStep/panel_A_condition_number.py` | Alignment vs condition number, online learning |
| Figure 4 A | `Figure4/SingleStep/panel_A_batch_size.py` | Alignment vs batch size, with covariance rescaling |
