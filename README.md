# Bible-Interpreter
Can translate the Bible in basic English, and concise.





## Jetson Container Setup
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
jetson-containers run -v /home/nvidia/Project:/workspace $(autotag transformers)
cd /workspace
python3 bible_inference.py
