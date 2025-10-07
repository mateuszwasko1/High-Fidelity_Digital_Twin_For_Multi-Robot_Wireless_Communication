# High-Fidelity_Digital_Twin_For_Multi-Robot_Wireless_Communication

# 🧩 Conda Environment Commands Cheat Sheet

# 1️⃣ Install a dependency
conda install -c conda-forge <package-name> -y

# Example:
conda install -c conda-forge matplotlib -y


# 2️⃣ Export the environment (after adding new packages)
conda env export --name bullet39 --no-builds | grep -v "^prefix: " > environment.yml


# 3️⃣ Start / activate the virtual environment
conda activate bullet39


# 4️⃣ Exit / deactivate the virtual environment
conda deactivate

# Recreate the enviroment
conda env create -f environment.yml

# Update the enviroment
conda env update -n bullet39 --file environment.yml --prune