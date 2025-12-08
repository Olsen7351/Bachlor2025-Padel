# 1. Auto-detect shots

python "01 Annotate.py" auto_annotate --video "videos/your_video.mp4"

# 2. Fix mistakes

python "01 Annotate.py" validate

# 3. Generate training data (select correct player)

python "01 Annotate.py" prepare

# 4. Train

# 1. Auto-detect shots

python "01 Annotate.py" auto_annotate --video "videos/your_video.mp4"

# 2. Fix mistakes

python "01 Annotate.py" validate

# 3. Generate training data (select correct player)

python "01 Annotate.py" prepare

# 3 Verify

python "01 Annotate.py" verify_final

# 4. Train

python "02 Train.py" --data dataset/player_enhanced
