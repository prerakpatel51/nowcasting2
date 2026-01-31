# Installation instructions of nowcasting library

1. Clone the repository
```
git clone https://github.com/FLOWARN/nowcasting.git
```

2. Navigate to the nowcasting directory
```
cd nowcasting2/
```

3. Create conda environment from the tito_env.yml file
```
conda env create -f tito_env.yml
```

4. Activate the conda environment
```
conda activate tito_env
```

5. Install pip packages from the requirements file (if using orchestrator)
```
pip install -r requirements.txt
```

6. Install the servir package locally
```
pip install -e .
```
