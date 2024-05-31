import numpy as np
import argparse
from PIL import Image

from Simulator.FRCN_Simulator import FRCN_Simulator





def main(parser):
    global args
    args, unknown = parser.parse_known_args()
    optimized_components = {"PM_h":args.PMH, "PM_v":args.PMV, "B_h":args.BH, "B_v":args.BV, "S_A":args.SA, "S_W":args.SW, "S_img":args.SIMG}
    frcn = FRCN_Simulator(IN_H=args.IN_H,
                          IN_W=args.IN_W,
                          optimized_components=optimized_components,
                          benchmark="201", benchmark_root="/Users/tomomasayamasaki/Library/CloudStorage/OneDrive-SingaporeUniversityofTechnologyandDesign/SUTD/Life_of_University/Lab/#4Research-RBFleX/NATS-Bench/NATS-tss-v1_0-3ffb9-simple",
                          hd_obj=2,
                          n_hyper=10, ref_score=-1000, acqu_algo="random", iters=50, n_init_size=100)

    """
    an image is loaded
    """
    image = Image.open(args.image_path)
    if not args.IN_H == image.height or args.IN_W == image.width:
        image = image.resize((args.IN_W, args.IN_H))
    image = np.array(image)


    """
    Run Simulator
    """
    frcn.run(image)


if __name__ == '__main__':
    """
    initial set parameters
    """
    parser = argparse.ArgumentParser(description='DL2FRCN_Simulator')
    parser.add_argument('-ih','--IN-H', default=224, type=int, help='Height of input image for faster RCNN (default: 224)')
    parser.add_argument('-iw','--IN-W', default=224, type=int, help='Width of input image for faster RCNN (default: 224)')
    parser.add_argument('--PMH', default=0, type=int)
    parser.add_argument('--PMV', default=0, type=int)
    parser.add_argument('--BH', default=0,type=int)
    parser.add_argument('--BV', default=0,type=int)
    parser.add_argument('--SA', default=0, type=int, help='buffer size (Byte)')
    parser.add_argument('--SW', default=0, type=int, help='buffer size (Byte)')
    parser.add_argument('--SIMG', default=0, type=int, help='buffer size (Byte)')
    parser.add_argument('-p','--image-path', required=True, help='path for an input image')
    main(parser)
