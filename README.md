# Setup

Run the following commands:
1. `git clone https://github.com/arianbakh/MTTS-CAN`
2. `cd MTTS-CAN`
3. `conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1`
4. `conda activate tf-gpu`
5. `pip install -r requirements.txt`

Tested on the following environments:
1. Ubuntu 20.04.6; Nvidia driver 470.256.02; CUDA 11.4 (outside conda env); conda 4.11.0; Python 3.9.7 (inside conda env)
2. Ubuntu 24.04; Nvidia driver 555.52.04; CUDA 12.5 (outside conda env); conda 23.1.0; Python 3.9.19 (inside conda env)

# Inference

Inference may freeze if you use GPU(s), in which case you have to set the `CUDA_VISIBLE_DEVICES` environment variable to null string when running the code:
1. `CUDA_VISIBLE_DEVICES= python code/predict_vitals.py --video_path videos/vid.mp4 --sample-dir outputs/sample.png --fig-dir outputs/fig.png --pulse-dir outputs/pulse.npy --resp-dir outputs/resp.npy --model-path mtts_can.hdf5`
