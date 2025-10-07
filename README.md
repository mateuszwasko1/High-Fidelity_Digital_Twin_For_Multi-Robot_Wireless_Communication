# High-Fidelity_Digital_Twin_For_Multi-Robot_Wireless_Communication

# Install a dependency

Try and install everything with Conda, but if not possible pip is fine
```bash
conda install -c conda-forge <package-name> -y
```

# Export the environment
When something changes in the environment (like you add a new dependency) you need to update the environment file by pasting the following
```bash
conda env export --from-history | grep -v "^prefix:" > environment.yml
```

# Start / activate the virtual environment
Before you start working, activate the virtual env:
```bash
conda activate bullet39
```

# Exit / deactivate the virtual environment
```bash
conda deactivate
```

# Recreate the environment
When recreating the env for the first time run the following:
```bash
conda env create -f environment.yml
```

# Update the environment
After pulling from git, ensure that your env is up to date:
```bash
conda env update -n bullet39 --file environment.yml --prune
```