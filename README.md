# Railway-Track-defect-detection

## Datasets
Dataset1 - https://www.kaggle.com/datasets/ashikadnan/railway-track-fault-detection-dataset2fastener

Dataset2 - https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection

Dataset3 - https://www.kaggle.com/datasets/ashikadnan/railway-track-fault-detection-dataset1-rail

### Setup kaggle API

For mac/linux

1. Create the .kaggle directory in your home folder
mkdir -p ~/.kaggle

2. Move kaggle.json into the directory
mv ./kaggle.json ~/.kaggle/

3. Set proper permissions (read/write only for your user)
chmod 600 ~/.kaggle/kaggle.json

For Windows(Powershell)

1. Create .kaggle directory in your user profile
mkdir $env:USERPROFILE\.kaggle

2. Move kaggle.json into the directory
move .\kaggle.json $env:USERPROFILE\.kaggle\

3. (Optional) Ensure it's not accessible by other users
Windows usually handles this automatically
