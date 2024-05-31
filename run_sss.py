import argparse

from Simulator.FRCN_Simulator import FRCN_Simulator




def main(parser):
    global args
    args, unknown = parser.parse_known_args()
    optimized_components = {"X1":0, "X2":0, "X3":0, "X4":0, "X5":0, "X6":0}
    frcn = FRCN_Simulator(IN_H=args.IN_H,
                          IN_W=args.IN_W,
                          optimized_components=optimized_components,
                          n_hyper=10, ref_score=-1000, acqu_algo="qNEHVI", iters=15, n_init_size=100,
                          Hardware_Arch="DeFiNES", mapping="os")  #ws, os, is / ScaleSim, DeFiNES, DL2

    """
    Run Simulator
    """
    frcn.run()


if __name__ == '__main__':
    """
    initial set parameters
    """
    parser = argparse.ArgumentParser(description='DL2FRCN_Simulator')
    parser.add_argument('-ih','--IN-H', default=32, type=int, help='Height of input image for faster RCNN (default: 224)')
    parser.add_argument('-iw','--IN-W', default=32, type=int, help='Width of input image for faster RCNN (default: 224)')
    main(parser)
