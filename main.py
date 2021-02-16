
""" 
Main script.
"""

import models
import argparse
from datetime import datetime as dt


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to configuration file')
    ap.add_argument('-d', '--data-root', required=True, type=str, help='Path to directory where data is saved as data.txt')
    ap.add_argument('-r', '--test-root', type=str, help='Path to directory where test data is saved as test_data.txt')
    ap.add_argument('-o', '--output', default=dt.now().strftime("%Y-%m-%d_%H-%M"), type=str, help='Path to output directory')
    ap.add_argument('-t', '--task', default='mlm', type=str, help='Task to perform, choose between (mlm,)')
    ap.add_argument('-l', '--load', type=str, help='Path to directory from which best_model.ckpt should be loaded')
    args = vars(ap.parse_args())

    # Initialize model
    if args['task'] == 'aec':
        trainer = models.AdverseEventClassification(args)
        trainer.train()

    elif args['task'] == 'embeds' and args['load'] is not None:
        trainer = models.AdverseEventClassification(args)
        trainer.save_embeds_for_projection()

    elif args['task'] == 'test' and args['load'] is not None and args['test_root'] is not None:
        trainer = models.AdverseEventClassification(args)
        trainer.test_model()
    
    else:
        raise ValueError(f"Unrecognized task {args['task']}")
