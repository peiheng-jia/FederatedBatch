# GitHub Repository Setup

This folder is prepared for publishing as the FederatedBatch release repository. Run the commands below from the project root.

Project root:

```powershell
cd D:\code\publication_release
```

## 1. Verify Git

Verify that Git is available:

```powershell
git --version
```

## 2. Review What Should Go To GitHub

Recommended to include:

- `src/`
- `experiments/`
- `scripts/`
- `visualization/`
- `manuscript/`
- `docs/`
- `MANUSCRIPT_FIGURE_TEXTS/`
- `README.md`
- `requirements.txt`
- `environment_r_viz.yml`
- `setup.py`
- `LICENSE`
- focused project documentation

Recommended to exclude:

- raw datasets such as `*.h5ad`, `*.h5`, `*.hdf5`
- generated experiment folders under `results/`
- backup folders such as `results_backup_*`
- model checkpoints such as `*.pt`, `*.pth`
- local run folders such as `save/`
- generated figures under `figures_pre/` and `figures_v2/`
- temporary files, logs, and local environment files

The `.gitignore` file has been updated for these rules.

## 3. Initialize Local Git Repository

```powershell
git init
git branch -M main
git status
```

Check the `git status` output before adding files. Large data files, model checkpoints, and generated result directories should not appear. If they do, stop and update `.gitignore` before committing.

## 4. Make The First Commit

```powershell
git add .
git status
git commit -m "Initial project commit"
```

## 5. Create The GitHub Repository

On GitHub:

1. Go to https://github.com/new
2. Repository name: `FederatedBatch-Paper`
3. Choose Public or Private.
4. Do not initialize with README, `.gitignore`, or license because this project already has them.
5. Create repository.

## 6. Connect And Push

Replace `YOUR_USERNAME` with your GitHub username or organization.

```powershell
git remote add origin https://github.com/YOUR_USERNAME/FederatedBatch-Paper.git
git push -u origin main
```

## 7. Optional: Use Git LFS For Large Files

If you intentionally need to version selected large files, use Git LFS instead of normal Git.

```powershell
git lfs install
git lfs track "*.h5ad"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Track large files with Git LFS"
```

Only use this for files that must be shared through GitHub. For paper reproducibility, it is usually better to provide download scripts or external dataset links instead of committing raw datasets.

## 8. Basic Daily Workflow

```powershell
git status
git add path\to\changed_file.py
git commit -m "Describe the change"
git push
```

Before every push, run:

```powershell
git status
```

Make sure no raw data, checkpoints, or generated result directories are staged.
