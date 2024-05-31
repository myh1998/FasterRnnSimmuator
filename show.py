# python show.py -p "test_image.jpg"

import numpy as np
import argparse

from Simulator.FRCN_Simulator import FRCN_Simulator


def main(parser):
    global args
    args, unknown = parser.parse_known_args()
    #optimized_components = {"PM_h":args.PMH, "PM_v":args.PMV, "B_h":args.BH, "B_v":args.BV, "S_A":args.SA, "S_W":args.SW, "S_img":args.SIMG}
    optimized_components = {"X1":0, "X2":0, "X3":0, "X4":0, "X5":0, "X6":0}
    frcn = FRCN_Simulator(IN_H=args.IN_H,
                          IN_W=args.IN_W,
                          score_input_batch = 8,
                          optimized_components=optimized_components,
                          n_hyper=2, ref_score=-1000, acqu_algo="random", iters=1, n_init_size=2,
                          Hardware_Arch="ScaleSim", mapping="ws")


    """
    Run Simulator
    """
    frcn.run()


if __name__ == '__main__':
    """
    initial set parameters
    """
    parser = argparse.ArgumentParser(description='DL2FRCN_Simulator')
    parser.add_argument('-ih','--IN-H', default=224, type=int, help='Height of input image for faster RCNN (default: 224)')
    parser.add_argument('-iw','--IN-W', default=224, type=int, help='Width of input image for faster RCNN (default: 224)')
    main(parser)
