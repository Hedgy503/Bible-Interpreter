# Run Instructions

On line 16 in bible_inference.py replace the Prompt with whatever Bible verse you want to summarise.

Then (from inside the orin nano), run the following commands to enter the jetson dev coontainer:

'''
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
jetson-containers run -v /home/nvidia/Project:/workspace $(autotag transformers)
cd /workspace
python3 bible_inference.py
'''

Once here, run:

'''
python3 bible_inference.py
'''

This will summarize the verse you put in your code.
