Prerequisites:

Python 3
CPU or NVIDIA GPU + CUDA CuDNN
tensorflow 1.15.0
keras 2.3.0
dtcwt 0.12.0


Testing

python ./predict.py -dataroot ./your_dataroot -datatype datatype -predictpath ./output_path -batch_size batchsize

*datatype default: tif, jpg ,png

Example:

python ./predict.py -dataroot ./testImg -predictpath ./p -batch_size 3
python ./predict.py -dataroot ./testImg -datatype tif -predictpath ./p -batch_size 3
