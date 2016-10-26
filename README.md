CS395T Deep Learning Project

For setup installation:
1) Download and install Anaconda2 from the link:
   https://www.continuum.io/downloads#linux

2) For opencv, run the command:
   conda install -c menpo opencv=2.4.11

Converting Video to sketch dataset

1) Convert video to frames:
   bash get_frames.sh <video_file_name.mp4> <dir>

2) Retreive frames that matches with the template images
   python match.py <input_folder> <output_folder>

3) Convert the retrieved frames to sketch images
   copy batch-cart2sketch.scm to .gimp-2.8/scripts/ folder.
   run bash gimp-cartoon2sketch.sh <input_folder> <threshold>


